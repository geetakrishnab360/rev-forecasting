from prophet import Prophet
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
import snowflake.connector as connector
from datetime import datetime
import pytz

np.float_ = np.float64


def create_snowflake_connection():
    con = connector.connect(
        user='forecasting_app_service_account',
        account='c2gpartners.us-east-1',
        password='B!_ForecastApp2024',
        warehouse='POWERHOUSE',
    )
    return con


def update_experiments_data_to_db(exp_id, bu):
    try:
        conn = create_snowflake_connection()
        cur = conn.cursor()
        sql_query = f"""
        UPDATE DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.EXPERIEMENTS
        SET STATUS = 'completed', END_DATE = '{datetime.now().astimezone(pytz.timezone('GMT'))}'
        WHERE EXPERIMENT_ID = '{exp_id}' and BUSINESS_UNIT = '{bu}'
        """
        cur.execute(sql_query)
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)
        pass


def insert_model_results_to_db(user, request_id, bu, dictionary):
    try:
        conn = create_snowflake_connection()
        cur = conn.cursor()
        sql_query = f"""
        INSERT INTO DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.MODEL_METRICS
        (EXPERIMENT_ID, 
        BUSINESS_UNIT, 
        USER, 
        FEATURES, 
        HYPERPARAMETERS, 
        FINAL_SCORE, 
        AVG_TRAIN_MAPE, 
        AVG_TRAIN_MAPE_DEV,
        AVG_TRAIN_CONTRIBUTION_DEV,
        AVG_TEST_MAPE,
        AVG_TEST_MAPE_DEV,
        AVG_GROWTH_ERROR)
        VALUES
        ('{request_id}', 
        '{bu}', 
        '{user}', 
        '{dictionary['features']}', 
        '{dictionary['yo']}',
        '{dictionary['final_score']}', 
        '{dictionary['avg_train_mape']}', 
        '{dictionary['avg_train_mape_dev']}',
        '{dictionary['avg_train_contribution_dev']}', 
        '{dictionary['avg_test_mape']}', 
        '{dictionary['avg_test_mape_dev']}',
        '{dictionary['avg_growth_error']}')
        """
        cur.execute(sql_query)
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)
        pass


def insert_forecast_results_to_db(request_id, bu, user, model_name, forecast_date, forecast):
    try:
        conn = create_snowflake_connection()
        cur = conn.cursor()
        sql_query = f"""
        INSERT INTO DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.FORECASTS VALUES
        """
        values = []
        for i in range(len(forecast)):
            forecast_ds = forecast['ds'].iloc[i]
            forecast_value = forecast['y'].iloc[i]
            values.append(
                f"('{request_id}', '{bu}', '{user}', '{model_name}', '{forecast_date}', '{forecast_ds}', {forecast_value})")
        sql_query += ", ".join(values)
        cur.execute(sql_query)
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)
        pass


def mom_growth(y):
    return 1 - y.shift(1) / y


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    return 1 - mean_squared_error(y_true, y_pred) / np.var(y_true)


def initialize_prophet_model(disable_seasonalities=True, growth='flat'):
    model = Prophet(growth=growth)
    if disable_seasonalities:
        model.daily_seasonality = False
        model.weekly_seasonality = False
        model.monthly_seasonality = False
        model.yearly_seasonality = False
    return model


def calculate_metrics(act, pred):
    merged = act[['ds', 'y']].merge(pred, on='ds')

    # mape
    merged['mape'] = 100 * abs(1 - merged['yhat'] / merged['y'])
    mape = merged['mape'].mean()
    mape_std = merged['mape'].std()

    # contributions
    pred_ = pred.drop(['ds', 'yhat'], axis=1).abs()
    base_contribution = (pred_[['trend', 'yearly']].sum(axis=1) / pred_.sum(axis=1)).mean()
    feature_contributions = (pred_.drop(['trend', 'yearly'], axis=1).div(pred_.sum(axis=1), axis=0)).mean().to_dict()
    feature_contributions.update({'base': base_contribution})
    contribution_std = np.std(list(feature_contributions.values()))
    return mape, mape_std, feature_contributions, contribution_std


def analyze_results(result_df):
    result_df['train_score'] = (
            (result_df['avg_train_mape'] / result_df['avg_test_mape'] > 10) & result_df['avg_train_mape'] >= 5).map(
        {True: 10, False: 0})
    for col in ['avg_test_mape', 'avg_test_mape_dev', 'avg_growth_error']:
        power = np.ceil(np.log(result_df[col].quantile(0.25)) / np.log(10))
        result_df[f'{col}_standardized'] = result_df[col] / 10 ** power
    result_df['final_score'] = result_df['train_score'] + result_df['avg_train_contribution_dev'] + \
                               result_df['avg_test_mape_standardized']
    columns = result_df.drop('final_score', axis=1).columns
    return result_df[['final_score', *columns]].sort_values(by='final_score')


def fit_model(data, testing_dates, prediction_date, feature_space=[[]], hyperparemeter_space=[], growth='linear',
              events=None):
    results = []
    for features in feature_space:
        for yo in hyperparemeter_space:
            train_mapes = []
            train_mape_stand_devs = []
            train_feature_contributions = []
            train_contribution_stand_devs = []
            test_mapes = []
            test_mape_stand_devs = []
            growth_errors = []
            for test_date in testing_dates:
                train = data[data.ds < test_date].copy()
                test = data[(data.ds >= test_date) & (data.ds < prediction_date)].copy()
                future = data[data.ds >= prediction_date - pd.DateOffset(months=1)]
                model = initialize_prophet_model(growth=growth)
                model.holidays = events
                model.add_seasonality('yearly', period=365.25, fourier_order=yo)
                for f in features:
                    model.add_regressor(f)
                model.fit(train)

                # train
                train_pred = model.predict(train)
                train_mape, train_mape_std, feature_contributions, contribution_std = calculate_metrics(train,
                                                                                                        train_pred[
                                                                                                            ['ds',
                                                                                                             'yhat',
                                                                                                             'trend',
                                                                                                             'yearly',
                                                                                                             *features]]
                                                                                                        )
                train_mapes.append(train_mape)
                train_mape_stand_devs.append(train_mape_std)
                train_feature_contributions.append(feature_contributions)
                train_contribution_stand_devs.append(contribution_std)

                # test
                test_pred = model.predict(test)
                test_mape, test_mape_std, _, _ = calculate_metrics(test, test_pred[
                    ['ds', 'yhat', 'trend', 'yearly', *features]])
                test_mapes.append(test_mape)
                test_mape_stand_devs.append(test_mape_std)

                # future
                future_pred = model.predict(future)
                future_growth = mom_growth(future_pred['yhat'])

                # past-1-year growth
                past_one_year_growth = np.array([])
                past_one_year_dates = [d - pd.DateOffset(years=1) for d in future['ds']]
                past_one_year = data[data.ds.isin(past_one_year_dates)].copy()
                if len(past_one_year_dates) == len(past_one_year):
                    past_one_year_growth = mom_growth(past_one_year['y']).values

                # 2 years before growth
                past_two_year_growth = np.array([])
                past_two_year_dates = [d - pd.DateOffset(years=2) for d in future['ds']]
                past_two_year = data[data.ds.isin(past_two_year_dates)].copy()
                if len(past_two_year_dates) == len(past_two_year):
                    past_two_year_growth = mom_growth(past_two_year['y']).values

                avg_past_growth = (past_two_year_growth + past_one_year_growth) / 2

                if len(avg_past_growth) == 0:
                    growth_error = 0.
                else:
                    # # print(avg_past_growth, future_growth)
                    # # growth_error = np.sqrt(mean_squared_error(avg_past_growth[1:], future_growth[1:]))
                    # growth_error = 0.
                    # print(avg_past_growth, future_growth)
                    growth_error = np.sqrt(mean_squared_error(avg_past_growth[1:], future_growth[1:]))

                growth_errors.append(growth_error)

            results.append({
                'features': features,
                'yo': yo,
                'avg_train_mape': np.mean(train_mapes),
                'avg_train_mape_dev': np.mean(train_mape_stand_devs),
                'avg_train_contribution_dev': np.mean(train_contribution_stand_devs),
                'avg_test_mape': np.mean(test_mapes),
                'avg_test_mape_dev': np.mean(test_mape_stand_devs),
                'avg_growth_error': np.mean(growth_errors),
                'train_mapes': train_mapes,
                'train_mape_stand_devs': train_mape_stand_devs,
                'train_feature_contributions': train_feature_contributions,
                'train_contribution_stand_devs': train_contribution_stand_devs,
                'test_mapes': test_mapes,
                'test_mape_stand_devs': test_mape_stand_devs,
                'growth_errors': growth_errors,
            })
    return pd.DataFrame(results)


def lambda_handler(event, context):
    ## TRAINING
    user = event['user']
    model_name = event['model_name']
    if event['holiday_dates_dict']:
        holiday_dates_dict = event['holiday_dates_dict']
        holiday_dates = pd.DataFrame(holiday_dates_dict)
        holiday_dates['ds'] = pd.to_datetime(holiday_dates['ds'])
    else:
        holiday_dates = None
    bu = event['bu']
    final_df_dict = event['final_df_dict']
    final_df = pd.DataFrame(final_df_dict)
    final_df['ds'] = pd.to_datetime(final_df['ds'])
    test_dates = event['test_dates']
    request_id = event['request_id']
    growth = 'linear'
    if bu == 'fpai':
        growth='flat'
    results = fit_model(final_df,
                        [pd.to_datetime(d) for d in test_dates],
                        pd.to_datetime('2024-09-01'),
                        hyperparemeter_space=range(1, 15),
                        events=holiday_dates,
                        growth=growth)
    results = analyze_results(results).reset_index(drop=True)
    best_model_dict = results.iloc[0].to_dict()
    res_dict = {'BU': bu, 'request_id': request_id, 'results': best_model_dict}
    print(res_dict)
    print(best_model_dict)
    update_experiments_data_to_db(request_id,
                                  bu)
    insert_model_results_to_db(user,
                               request_id,
                               bu,
                               best_model_dict)
    print("Training completed")

    ## Future Predictions
    print("Starting Future Predictions")
    test_date = pd.to_datetime('2024-06-01')
    prediction_date = pd.to_datetime('2024-09-01')
    prediction_df = pd.date_range('2025-09-01', '2025-12-01', freq='MS', name='ds').to_frame().reset_index(drop=True)
    final_df = pd.concat((final_df, prediction_df))

    data = final_df.copy()
    data = data.reset_index(drop=True)

    train = data[data.ds < test_date].copy()
    future = data[data.ds >= prediction_date].reset_index(drop=True)
    future_model = initialize_prophet_model(growth='linear')
    future_model.holidays = holiday_dates
    future_model.add_seasonality('yearly', period=365.25, fourier_order=best_model_dict['yo'])
    if best_model_dict['features']:
        for f in best_model_dict['features']:
            future_model.add_regressor(f)
    future_pred = future_model.fit(train).predict(future)
    print("Future Predictions completed")
    future_res = future_pred[['ds', 'yhat']]
    if bu == 'bts':
        start_date = '2019-01-01'
        end_date = '2025-12-31'
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Define the US Federal Holiday Calendar
        us_cal = USFederalHolidayCalendar()

        # Get all the holidays between 2019 and 2025
        holidays = us_cal.holidays(start=start_date, end=end_date)

        # Filter out weekends and holidays to get working days
        business_days = dates[(dates.weekday < 5) & (~dates.isin(holidays))]
        # Create a dataframe to group by year and month, counting working days
        df = pd.DataFrame({'Date': business_days})
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        working_days_df = df.groupby(['Year', 'Month']).count().reset_index().rename(columns=
                                                                                     {'Date': 'total_working_dayss',
                                                                                      'Year': 'year',
                                                                                      'Month': 'month'
                                                                                      })
        future = future.merge(working_days_df, left_on=[future.ds.dt.year, future.ds.dt.month],
                              right_on=['year', 'month'], how='left')
        future_res['yhat'] = future_res['yhat'] * future['total_working_dayss']
    future_res.rename(columns={'yhat': 'y'}, inplace=True)
    forecast_date = datetime.now().astimezone(pytz.timezone('GMT'))
    insert_forecast_results_to_db(request_id,
                                  bu,
                                  user,
                                  model_name,
                                  forecast_date,
                                  future_res)
    print("Forecast results inserted to DB")
    return {
        'statusCode': 200,
    }