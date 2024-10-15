import numpy as np

np.float_ = np.float64
import pandas as pd
from pathlib import Path
from random import choice
from datetime import datetime
from prophet import Prophet
from sklearn.metrics import mean_squared_error


def drop_null_rows_columns(input_df):
    _df = input_df.copy()
    return _df.dropna(how="all").dropna(how='all', axis=1).reset_index(drop=True)


def set_column_header(input_df, header_index=0):
    _df = input_df.copy()
    return _df.rename(columns=_df.iloc[header_index]).drop(header_index)


def convert_to_long_form(input_df):
    _df = input_df.copy()
    return _df.set_index(_df.columns[0]).iloc[:, :12].stack(level=0).reset_index()


def create_date_column(input_df):
    _df = input_df.copy()
    column_name = _df.columns[0]
    _df['ds'] = (_df[column_name].astype('str') + _df['level_1']).apply(lambda x: datetime.strptime(x, '%Y%b'))
    return _df.rename(columns={0: 'y'}).drop([column_name, 'level_1'], axis=1)


def convert_output_to_floats(input_df):
    _df = input_df.copy()
    _df['y'] = _df['y'].astype('float')
    return _df


def preprocess_file_version1(filename):
    # actual_df = pd.read_excel(Path("data") / Path(f"{filename}.xlsx"))
    actual_df = pd.read_excel(filename)
    null_indexes = actual_df[actual_df.isnull().all(1)].index
    slices = [slice(0, idx) if i == 0 else slice(null_indexes[i - 1] + 1, idx) for i, idx in enumerate(null_indexes)]
    slices.append(slice(null_indexes[-1], None))
    df_list = [actual_df[slice_] for slice_ in slices]
    df_list = list(filter(lambda x: x.dropna(how='all').shape[0] > 0, df_list))
    df_list = [df_.pipe(drop_null_rows_columns) \
                   .pipe(set_column_header) \
                   .pipe(convert_to_long_form) \
                   .pipe(create_date_column) \
                   .pipe(convert_output_to_floats)
               for df_ in df_list]
    return {'bts': df_list[0],
            'dsx': df_list[1],
            'emea': df_list[2],
            'mlabs': df_list[3],
            'fpai': df_list[4]
            }


def initialize_prophet_model(disable_seasonalities=True, growth='flat'):
    model = Prophet(growth=growth)
    if disable_seasonalities:
        model.daily_seasonality = False
        model.weekly_seasonality = False
        model.monthly_seasonality = False
        model.yearly_seasonality = False
    return model


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
                # 'future_predictions' : future_pred.iloc[1:],
                # 'train_predictions'
            })
    return pd.DataFrame(results)


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
    # + result_df['avg_growth_error_standardized']
    columns = result_df.drop('final_score', axis=1).columns
    return result_df[['final_score', *columns]].sort_values(by='final_score')


def mom_growth(y):
    return 1 - y.shift(1) / y


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


def get_random_features(feature_dic, n_iterations=100):
    feature_space = []
    choice_set = set()
    for i in range(n_iterations):
        columns = []
        for f, t_list in feature_dic.items():
            t = choice(t_list)
            if t > 0:
                columns.append(f'{f}_lag_{t}')
            if t == 0:
                columns.append(f)
        hash_ = hash(''.join(columns))
        if hash_ not in choice_set:
            feature_space.append(columns.copy())
            choice_set.add(hash_)
    return feature_space
