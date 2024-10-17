from prophet import Prophet
from functools import reduce
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


def fetch_revenue_data_from_db():
    print("[INFO] fetching data from SNOWFLAKE")
    sql_query = f"""
    SELECT
        * 
    FROM DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.REVENUE_DATA
    """.strip()

    try:
        conn = create_snowflake_connection(
        )
        cur = conn.cursor()
        cur.execute(sql_query)
        data = pd.DataFrame(
            cur.fetchall(), columns=[desc[0] for desc in cur.description]
        )
        conn.close()
        return data
    except Exception as e:
        print(e)
        pass


def preprocess(
        dataframe,
):
    def convert_columns_to_lower(df):
        df.columns = [col.lower() for col in df.columns]
        return df

    def convert_to_dates(df):
        df['date'] = pd.to_datetime(df['date'])
        return df

    def convert_to_float(df):
        df['revenue'] = df['revenue'].astype(float)
        return df

    def rename_columns(df):
        df.rename(columns={'date': 'ds', 'revenue': 'y'}, inplace=True)
        return df

    def trim_date(df):
        df['ds'] = df.ds.apply(lambda x: datetime(x.year, x.month, 1))
        return df

    return reduce(
        lambda df, func: func(df),
        [
            convert_columns_to_lower,
            convert_to_dates,
            convert_to_float,
            rename_columns,
            trim_date,
        ],
        dataframe,
    )


def dsx_method(rev_data, exo_df):
    bu = 'dsx'
    dsx_df = rev_data[rev_data['business_unit'] == bu]
    actuals_latest_date_time = dsx_df.ds.max()
    prediction_df = pd.date_range((actuals_latest_date_time + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
                                  (actuals_latest_date_time + pd.DateOffset(months=12)).strftime("%Y-%m-%d"), freq='MS',
                                  name='ds').to_frame().reset_index(drop=True)
    final_df = pd.concat((dsx_df, prediction_df))
    final_df = final_df[final_df.ds >= '2020-07-01']
    final_df = final_df.reset_index(drop=True)
    final_df = final_df.merge(exo_df, on='ds', how='left')

    holiday_dates = pd.DataFrame({
        'holiday': ['NA'] * 3,
        'ds': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
        'lower_window': 0,
        'upper_window': 1
    })

    test_dates = [(actuals_latest_date_time - pd.DateOffset(months=2)).strftime("%Y-%m-%d"),
                  (actuals_latest_date_time - pd.DateOffset(months=4)).strftime("%Y-%m-%d")]

    return holiday_dates, final_df, test_dates


def bts_method(rev_data, exo_df):
    holiday_dates = None
    bu = 'bts'
    bts_df = rev_data[rev_data['business_unit'] == bu]
    actuals_latest_date_time = bts_df.ds.max()
    prediction_df = pd.date_range((actuals_latest_date_time + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
                                  (actuals_latest_date_time + pd.DateOffset(months=12)).strftime("%Y-%m-%d"), freq='MS',
                                  name='ds').to_frame().reset_index(drop=True)
    final_df = pd.concat((bts_df, prediction_df))
    final_df = final_df[final_df.ds >= '2021-01-01']
    final_df = final_df.reset_index(drop=True)
    start_date = '2019-01-01'
    end_date = (actuals_latest_date_time + pd.DateOffset(months=24))
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
                                                                                 {'Date': 'total_working_days',
                                                                                  'Year': 'year',
                                                                                  'Month': 'month'
                                                                                  })
    final_df['month'] = final_df.ds.dt.month
    final_df['year'] = final_df.ds.dt.year
    final_df = final_df.merge(working_days_df, on=['year', 'month'], how='left').copy()
    final_df['y_monthly'] = final_df['y']
    final_df['y'] = final_df['y_monthly'] / final_df['total_working_days']
    test_dates = [(actuals_latest_date_time - pd.DateOffset(months=2)).strftime("%Y-%m-%d"),
                  (actuals_latest_date_time - pd.DateOffset(months=4)).strftime("%Y-%m-%d")]
    return holiday_dates, final_df, test_dates


def emea_method(rev_data, exo_df):
    bu = 'emea'
    emea_df = rev_data[rev_data['business_unit'] == bu]
    actuals_latest_date_time = emea_df.ds.max()
    prediction_df = pd.date_range((actuals_latest_date_time + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
                                  (actuals_latest_date_time + pd.DateOffset(months=12)).strftime("%Y-%m-%d"), freq='MS',
                                  name='ds').to_frame().reset_index(drop=True)
    final_df = pd.concat((emea_df, prediction_df))
    final_df = final_df[final_df.ds >= '2021-01-01']
    final_df = final_df.reset_index(drop=True)
    final_df = final_df.merge(exo_df, on='ds', how='left')

    holiday_dates = pd.DataFrame({
        'holiday': ['NA'],
        'ds': pd.to_datetime(['2021-12-01']),
        'lower_window': 0,
        'upper_window': 1
    })
    test_dates = [(actuals_latest_date_time - pd.DateOffset(months=2)).strftime("%Y-%m-%d"),
                  (actuals_latest_date_time - pd.DateOffset(months=4)).strftime("%Y-%m-%d")]

    return holiday_dates, final_df, test_dates


def mlabs_method(rev_data, exo_df):
    bu = 'mlabs'
    mlabs_df = rev_data[rev_data['business_unit'] == bu]
    actuals_latest_date_time = mlabs_df.ds.max()
    prediction_df = pd.date_range((actuals_latest_date_time + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
                                  (actuals_latest_date_time + pd.DateOffset(months=12)).strftime("%Y-%m-%d"), freq='MS',
                                  name='ds').to_frame().reset_index(drop=True)
    final_df = pd.concat((mlabs_df, prediction_df))
    final_df = final_df[final_df.ds >= '2022-01-01']
    final_df = final_df.reset_index(drop=True)
    final_df = final_df.merge(exo_df, on='ds', how='left')

    holiday_dates = None

    test_dates = [(actuals_latest_date_time - pd.DateOffset(months=2)).strftime("%Y-%m-%d"),
                  (actuals_latest_date_time - pd.DateOffset(months=4)).strftime("%Y-%m-%d")]

    return holiday_dates, final_df, test_dates


def fpai_method(rev_data, exo_df):
    bu = 'fpai'
    fpai_df = rev_data[rev_data['business_unit'] == bu]
    actuals_latest_date_time = fpai_df.ds.max()
    prediction_df = pd.date_range((actuals_latest_date_time + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
                                  (actuals_latest_date_time + pd.DateOffset(months=12)).strftime("%Y-%m-%d"), freq='MS',
                                  name='ds').to_frame().reset_index(drop=True)
    final_df = pd.concat((fpai_df, prediction_df))
    final_df = final_df[final_df.ds >= '2023-01-01']
    final_df = final_df.reset_index(drop=True)
    final_df = final_df.merge(exo_df, on='ds', how='left')
    # bu = 'fpai'
    # fpai_df = rev_data[rev_data['business_unit'] == bu]
    # actuals_latest_date_time = rev_data.ds.max()
    # final_df = fpai_df.merge(exo_df, on='ds', how='right')
    # prediction_df = pd.date_range((actuals_latest_date_time + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
    #                               (actuals_latest_date_time + pd.DateOffset(months=12)).strftime("%Y-%m-%d"), freq='MS',
    #                               name='ds').to_frame().reset_index(drop=True)
    # final_df = pd.concat((final_df, prediction_df))
    # final_df = final_df[final_df.ds >= '2023-01-01']
    # final_df = final_df.reset_index(drop=True)
    # bu = 'fpai'
    # fpai_df = rev_data[rev_data['business_unit'] == bu]
    # actuals_latest_date_time = fpai_df.ds.max()
    # prediction_df = pd.date_range((actuals_latest_date_time + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
    #                               (actuals_latest_date_time + pd.DateOffset(months=12)).strftime("%Y-%m-%d"), freq='MS',
    #                               name='ds').to_frame().reset_index(drop=True)
    # final_df = pd.concat((fpai_df, prediction_df))
    # final_df = final_df.reset_index(drop=True)
    # final_df = final_df[final_df.ds >= '2023-01-01']
    # final_df = final_df.merge(exo_df, on='ds', how='left')
    start_date = '2019-01-01'
    end_date = (actuals_latest_date_time + pd.DateOffset(months=24))
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
                                                                                 {'Date': 'total_working_days',
                                                                                  'Year': 'year',
                                                                                  'Month': 'month'
                                                                                  })
    final_df['month'] = final_df.ds.dt.month
    final_df['year'] = final_df.ds.dt.year
    final_df = final_df.merge(working_days_df, on=['year', 'month'], how='left').copy()
    holiday_dates = pd.DataFrame({
        'holiday': ['NA'] * 3,
        'ds': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
        'lower_window': 0,
        'upper_window': 1
    })

    test_dates = [(actuals_latest_date_time - pd.DateOffset(months=2)).strftime("%Y-%m-%d"),
                  (actuals_latest_date_time - pd.DateOffset(months=4)).strftime("%Y-%m-%d")]

    return holiday_dates, final_df, test_dates


def insert_experiments_data_to_db(dictionary):
    try:
        conn = create_snowflake_connection()
        cur = conn.cursor()
        sql_query = f"""
        INSERT INTO DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.EXPERIEMENTS VALUES
        ('{dictionary['experiment_id']}',
        '{dictionary['business_unit']}', 
        '{dictionary['start_date']}',
         NULL, 
        '{dictionary['status']}', 
        '{dictionary['user']}', 
        NULL)
        """
        cur.execute(sql_query)
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)
        pass


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
        INSERT INTO DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.MODEL_METRICS VALUES
        ('{request_id}', 
        '{bu}', 
        '{user}', 
        '{",".join(dictionary['features'])}', 
        {dictionary['yo']},
        {dictionary['final_score']}, 
        {dictionary['avg_train_mape']}, 
        {dictionary['avg_train_mape_dev']},
        {dictionary['avg_train_contribution_dev']}, 
        {dictionary['avg_test_mape']}, 
        {dictionary['avg_test_mape_dev']},
        {dictionary['avg_growth_error']})
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


def create_exo_df_():
    """
        Creates exo dataframe from Exogenous data
    """
    csv_url = f'https://raw.githubusercontent.com/rohankblend/Forecasting_exog_files/master/exog_variables.csv'
    exo_df = pd.read_csv(csv_url, index_col=0, )

    exo_df.DATE = pd.to_datetime(exo_df.DATE)

    for col in exo_df.columns:
        if exo_df[col].dtype == 'object':
            exo_df[col] = exo_df[col].str.replace(',', '').astype(np.float64)

    exo_df['ds'] = exo_df.DATE.apply(lambda x: datetime(x.year, x.month, 1))
    exo_df = exo_df.drop('DATE', axis=1).fillna(0).groupby('ds').mean().reset_index()
    return exo_df


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


def fit_model_dsx_emea(data, testing_dates, prediction_date, feature_space=[[]], hyperparemeter_space=[],
                       growth='linear',
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


def fit_model_bts_mlabs_fpai(data, testing_dates, prediction_date, feature_space=[[]], hyperparemeter_space=[],
                             growth='linear',
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
                # print(len(past_one_year_dates),len(past_one_year))
                if len(past_one_year_dates) == len(past_one_year):
                    past_one_year_growth = mom_growth(past_one_year['y']).values

                # 2 years before growth
                past_two_year_growth = np.array([])
                past_two_year_dates = [d - pd.DateOffset(years=2) for d in future['ds']]
                past_two_year = data[data.ds.isin(past_two_year_dates)].copy()
                # print((past_two_year_dates),past_two_year.ds)
                # return
                if len(past_two_year_dates) == len(past_two_year):
                    past_two_year_growth = mom_growth(past_two_year['y']).values

                # avg_past_growth = (past_two_year_growth + past_one_year_growth) / 2
                #
                # if len(avg_past_growth) == 0:
                #     growth_error = 0.
                # else:
                #     # print(avg_past_growth, future_growth)
                #     # growth_error = np.sqrt(mean_squared_error(avg_past_growth[1:], future_growth[1:]))
                #     growth_error = 0.
                growth_error = 0.

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
                # 'future_predictions' : future_pred.iloc[1:],
                # 'train_predictions'
            })
    return pd.DataFrame(results)


# def fit_model_fpai(data, testing_dates, prediction_date, feature_space=[[]], hyperparemeter_space=[], growth='linear',
#               events=None):
#     results = []
#     for features in feature_space:
#         for yo in hyperparemeter_space:
#             train_mapes = []
#             train_mape_stand_devs = []
#             train_feature_contributions = []
#             train_contribution_stand_devs = []
#             test_mapes = []
#             test_mape_stand_devs = []
#             growth_errors = []
#             for test_date in testing_dates:
#                 train = data[data.ds < test_date].copy()
#                 test = data[(data.ds >= test_date) & (data.ds < prediction_date)].copy()
#                 future = data[data.ds >= prediction_date - pd.DateOffset(months=1)]
#                 model = initialize_prophet_model(growth=growth)
#                 model.holidays = events
#
#                 model.add_seasonality('yearly', period=365.25, fourier_order=yo)
#                 for f in features:
#                     model.add_regressor(f)
#                 model.fit(train)
#
#                 # train
#                 train_pred = model.predict(train)
#                 train_mape, train_mape_std, feature_contributions, contribution_std = calculate_metrics(train,
#                                                                                                         train_pred[
#                                                                                                             ['ds',
#                                                                                                              'yhat',
#                                                                                                              'trend',
#                                                                                                              'yearly',
#                                                                                                              *features]]
#                                                                                                         )
#                 train_mapes.append(train_mape)
#                 train_mape_stand_devs.append(train_mape_std)
#                 train_feature_contributions.append(feature_contributions)
#                 train_contribution_stand_devs.append(contribution_std)
#
#                 # test
#                 test_pred = model.predict(test)
#                 test_mape, test_mape_std, _, _ = calculate_metrics(test, test_pred[
#                     ['ds', 'yhat', 'trend', 'yearly', *features]])
#                 test_mapes.append(test_mape)
#                 test_mape_stand_devs.append(test_mape_std)
#
#                 # future
#                 future_pred = model.predict(future)
#                 future_growth = mom_growth(future_pred['yhat'])
#
#                 # past-1-year growth
#                 past_one_year_growth = np.array([])
#                 past_one_year_dates = [d - pd.DateOffset(years=1) for d in future['ds']]
#                 past_one_year = data[data.ds.isin(past_one_year_dates)].copy()
#                 # print(len(past_one_year_dates),len(past_one_year))
#                 if len(past_one_year_dates) == len(past_one_year):
#                     past_one_year_growth = mom_growth(past_one_year['y']).values
#
#                 # 2 years before growth
#                 past_two_year_growth = np.array([])
#                 past_two_year_dates = [d - pd.DateOffset(years=2) for d in future['ds']]
#                 past_two_year = data[data.ds.isin(past_two_year_dates)].copy()
#                 # print((past_two_year_dates),past_two_year.ds)
#                 # return
#                 if len(past_two_year_dates) == len(past_two_year):
#                     past_two_year_growth = mom_growth(past_two_year['y']).values
#
#                 # avg_past_growth = (past_two_year_growth + past_one_year_growth) / 2
#                 avg_past_growth = []
#                 if len(avg_past_growth) == 0:
#                     growth_error = 0.
#                 else:
#                     # print(avg_past_growth, future_growth)
#                     # growth_error = np.sqrt(mean_squared_error(avg_past_growth[1:], future_growth[1:]))
#                     growth_error = 0.
#
#                 growth_errors.append(growth_error)
#
#             results.append({
#                 'features': features,
#                 'yo': yo,
#                 'avg_train_mape': np.mean(train_mapes),
#                 'avg_train_mape_dev': np.mean(train_mape_stand_devs),
#                 'avg_train_contribution_dev': np.mean(train_contribution_stand_devs),
#                 'avg_test_mape': np.mean(test_mapes),
#                 'avg_test_mape_dev': np.mean(test_mape_stand_devs),
#                 'avg_growth_error': np.mean(growth_errors),
#                 'train_mapes': train_mapes,
#                 'train_mape_stand_devs': train_mape_stand_devs,
#                 'train_feature_contributions': train_feature_contributions,
#                 'train_contribution_stand_devs': train_contribution_stand_devs,
#                 'test_mapes': test_mapes,
#                 'test_mape_stand_devs': test_mape_stand_devs,
#                 'growth_errors': growth_errors,
#                 # 'future_predictions' : future_pred.iloc[1:],
#                 # 'train_predictions'
#             })
#     return pd.DataFrame(results)


def analyze_results_dsx_emea(result_df):
    result_df['train_score'] = (
            (result_df['avg_train_mape'] / result_df['avg_test_mape'] > 10) & result_df['avg_train_mape'] >= 5).map(
        {True: 10, False: 0})
    for col in ['avg_test_mape', 'avg_test_mape_dev', 'avg_growth_error']:
        power = np.ceil(np.log(result_df[col].quantile(0.25)) / np.log(10))
        result_df[f'{col}_standardized'] = result_df[col] / 10 ** power
    result_df['final_score'] = result_df['train_score'] + result_df['avg_train_contribution_dev'] + \
                               result_df['avg_test_mape_standardized']
    # + result_df['avg_growth_error_standardized']
    columns = result_df.drop('final_score', axis=1).columns
    return result_df[['final_score', *columns]].sort_values(by='final_score')


def analyze_results_bts_mlabs_fpai(result_df):
    result_df['train_score'] = (
            (result_df['avg_train_mape'] / result_df['avg_test_mape'] > 10) & result_df['avg_train_mape'] >= 5).map(
        {True: 10, False: 0})
    for col in ['avg_test_mape', 'avg_test_mape_dev']:  # removed avg_growth_error
        power = np.ceil(np.log(result_df[col].quantile(0.25)) / np.log(10))
        result_df[f'{col}_standardized'] = result_df[col] / 10 ** power
    result_df['final_score'] = result_df['train_score'] + result_df['avg_train_contribution_dev'] + \
                               result_df['avg_test_mape_standardized']
    # + result_df['avg_growth_error_standardized']
    columns = result_df.drop('final_score', axis=1).columns
    return result_df[['final_score', *columns]].sort_values(by='final_score')


def lambda_handler(event, context):
    bu = event['bu']
    print("BU: ", bu)
    user = event['user']
    model_name = event['model_name']
    request_id = event['request_id']

    model_info_dict = {
        'experiment_id': request_id,
        'business_unit': bu,
        'start_date': datetime.now().astimezone(pytz.timezone('GMT')),
        'status': 'initiated',
        'user': user,
        # 'failure_reason': None
    }
    insert_experiments_data_to_db(model_info_dict)

    exo_df = create_exo_df_()
    rev_data = fetch_revenue_data_from_db()
    rev_data = preprocess(rev_data)
    latest_actual_bu_date_time = rev_data[rev_data['business_unit'] == bu].ds.max()
    bu_methods_dict = {
        'dsx': [dsx_method, fit_model_dsx_emea, analyze_results_dsx_emea],
        'emea': [emea_method, fit_model_dsx_emea, analyze_results_dsx_emea],
        'bts': [bts_method, fit_model_bts_mlabs_fpai, analyze_results_bts_mlabs_fpai],
        'mlabs': [mlabs_method, fit_model_bts_mlabs_fpai, analyze_results_bts_mlabs_fpai],
        'fpai': [fpai_method, fit_model_bts_mlabs_fpai, analyze_results_bts_mlabs_fpai]
    }
    holiday_dates, final_df, test_dates = bu_methods_dict[bu][0](rev_data, exo_df)

    ## TRAINING
    growth = 'linear'
    features = []
    if bu == 'fpai':
        growth = 'flat'
        features = ['year']
    results = bu_methods_dict[bu][1](data=final_df,
                                     testing_dates=test_dates,
                                     prediction_date=latest_actual_bu_date_time + pd.DateOffset(months=1),
                                     hyperparemeter_space=range(1, 15),
                                     events=holiday_dates,
                                     feature_space=[features],
                                     growth=growth)
    results = bu_methods_dict[bu][2](results).reset_index(drop=True)
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
    test_date = latest_actual_bu_date_time - pd.DateOffset(months=2)
    prediction_date = latest_actual_bu_date_time + pd.DateOffset(months=1)

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
        future_res['yhat'] = future_res['yhat'] * future['total_working_days']

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
