import pandas as pd
from functools import reduce
import numpy as np
from collections import OrderedDict
import streamlit as st
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def prepare_actual_rev_data():
    actual_df = pd.read_excel(
        r"data/actuals.xlsx"
    )
    actual_df.columns = [
        "hubspot_id",
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
        "2024-04-01",
        "2024-05-01",
        "2024-06-01",
        "2024-07-01",
    ]
    actual_df = (
        actual_df.set_index("hubspot_id")[
            [
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
                "2024-05-01",
                "2024-06-01",
                "2024-07-01",
            ]
        ]
        .stack()
        .reset_index()
        .rename(columns={"level_1": "date", 0: "amount"})
    )
    actual_df.date = pd.to_datetime(actual_df.date).dt.date
    st.session_state["actual_df"] = actual_df.copy()
    # actuals = pd.read_excel('data/YTD 2024 Revenue.xlsx', sheet_name='Hubspot ID', skiprows=5, nrows=312)
    # actuals = actuals.rename(columns={'Entity (Line): HubSpot Deal ID': 'Hubspot ID'})
    # actuals = actuals.drop(columns=['Unnamed: 0'], axis=1)
    # actuals = actuals.drop(0)
    # actuals = actuals[actuals['Hubspot ID'] != '(blank)']
    # actuals = actuals.fillna(0)
    # drop_ids = ['15582551047, 16592648012',
    #             'MLabs-Embark-GenotypingPlatform',
    #             'MLabs-Embark-GenotypingPl',
    #             '14251479461; 16059181240',
    #             '14251479461; 16059181240',
    #             '17858842259; 17858935945']
    # actuals = actuals[~actuals['Hubspot ID'].isin(drop_ids)]
    # existing_actual_df = actuals[['Hubspot ID', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']]
    # existing_actual_df.columns = [
    #     "Hubspot ID",
    #     "2024-01-01",
    #     "2024-02-01",
    #     "2024-03-01",
    #     "2024-04-01",
    #     "2024-05-01",
    #     "2024-06-01",
    #     "2024-07-01",
    # ]
    # existing_actual_df = (
    #     existing_actual_df.set_index("Hubspot ID")[
    #         [
    #             "2024-01-01",
    #             "2024-02-01",
    #             "2024-03-01",
    #             "2024-04-01",
    #             "2024-05-01",
    #             "2024-06-01",
    #             "2024-07-01",
    #         ]
    #     ]
    #     .stack()
    #     .reset_index()
    #     .rename(columns={"level_1": "date", 0: "amount"})
    # )
    # existing_actual_df.date = pd.to_datetime(existing_actual_df.date).dt.date
    # st.session_state["actual_df"] = existing_actual_df.copy()


def convert_to_cohort_df(experiemnt_df):
    # print("******************Experiment df******************")
    # print(experiemnt_df)
    features = experiemnt_df["columns"].str.split(
        "|",
        expand=True,
        regex=False,
    )
    # print("******************Features******************")
    # print(features)
    features.columns = features.iloc[0]
    features = features.iloc[1:]
    features["eff_probability"] = (
        experiemnt_df["probabilities"].iloc[1:].copy()
    )
    features["selected"] = experiemnt_df["selected"].iloc[1:].copy()
    features = features.reset_index(drop=True)
    features["cohort"] = pd.Series(
        [f"cohort {i}" for i in range(0, len(experiemnt_df) - 1)]
    )
    # features.columns = features.iloc[0]
    # features.drop(0, inplace=True)
    return features


def preprocess(
        dataframe,
):
    def convert_columns_to_lower(df):
        df.columns = [col.lower() for col in df.columns]
        return df

    def convert_to_dates(df):
        date_columns = [
            "snapshot_date",
            "expected_project_start_date",
            "close_date",
            "create_date",
            "qualification_date",
            "est._project_end_date",
            "project_end_date",
        ]
        df[date_columns] = df[date_columns].transform(pd.to_datetime)
        return df

    def convert_to_float(df):
        numeric_columns = [
            "amount_in_company_currency",
            "expected_project_duration_in_months",
            "annual_contract_value",
            "total_contract_value",
            "effective_probability",
            "est._monthly_revenue_(company_currency)",
            "tcv_and_amount_delta",
            "deal_probability",
        ]
        df[numeric_columns] = df[numeric_columns].transform(
            pd.to_numeric, errors="coerce"
        )
        return df

    def strip_white_spaces(df):
        string_columns = df.select_dtypes(include="object").columns
        df[string_columns] = df[string_columns].transform(
            lambda x: x.str.strip()
        )
        return df

    def fill_nan_with_zero(df):
        columns = [
            "expected_project_duration_in_months",
            "amount_in_company_currency",
            "total_contract_value",
            "effective_probability",
            "tcv_and_amount_delta",
            "deal_probability",
        ]
        df[columns] = df[columns].fillna(0)
        return df

    def remove_wac(df):
        return df[df.work_ahead != "Yes"].copy()

    def remove_early_pipeline_data(df):
        return df[
            ~df.deal_stage.isin(
                ["0_new", "1_connected_to_meet", "2_needs_expressed"]
            )
        ].copy()

    def round_duration(df):
        df["expected_project_duration_in_months"] = np.round(
            df["expected_project_duration_in_months"]
        )
        return df

    def modify_start_date(x):
        if pd.isna(x):
            return x
        if x.day > 20:
            return x + pd.DateOffset(days=x.days_in_month - x.day + 1)
        else:
            return x - pd.DateOffset(days=x.day - 1)

    def add_modified_start_date(df):
        df["modified_start_date"] = df.expected_project_start_date.apply(
            modify_start_date
        )
        return df

    def add_snapshot_month(df):
        df["snapshot_month"] = df["snapshot_date"].dt.date
        return df

    def drop_na_start_dates(df):
        df = df[~df.expected_project_start_date.isna()]
        return df

    def remove_non_dsx(df):
        return df[
            df.pipeline.isin(
                [
                    "Blend360 DSX Pipeline",
                    "Blend360 BTS Pipeline",
                    "Blend360 Renewals",
                ]
            )
        ].copy()

    def create_final_probability(df):
        df["final_probability"] = df[
            ["deal_probability", "effective_probability"]
        ].max(axis=1)
        return df

    def drop_zero_contract_value_deals(df):
        df = df[df["total_contract_value"] != 0].copy()
        return df

    def drop_visa(df):
        # print(df.record_id.dtype)
        df = df[df.record_id != "17757257376"]
        return df

    preprocess_functions = [
        convert_columns_to_lower,
        convert_to_dates,
        convert_to_float,
        strip_white_spaces,
        fill_nan_with_zero,
        remove_wac,
        remove_early_pipeline_data,
        round_duration,
        add_modified_start_date,
        add_snapshot_month,
        create_final_probability,
        drop_visa,
    ]

    _df = reduce(lambda _df, f: _df.pipe(f), preprocess_functions, dataframe)
    hubspot_ids = _df.groupby("snapshot_month").record_id.unique().to_dict()
    return hubspot_ids, reduce(
        lambda _df, f: _df.pipe(f),
        [
            drop_na_start_dates,
            remove_non_dsx,
            drop_zero_contract_value_deals,
        ],
        _df,
    )


def _calculate_forecasts(
        df,
        probability_columns,
        amount_columns,
        forecast_start_month=None,
        forecast_end_month=None,
):
    _df = df.copy()
    _df["end_date"] = _df.apply(
        lambda x: x["modified_start_date"]
                  + pd.DateOffset(
            months=int(x["expected_project_duration_in_months"]) - 1
        ),
        axis=1,
    )

    output_df = pd.concat(
        _df.apply(
            expand_rows, axis=1, probability_columns=probability_columns
        ).tolist(),
        ignore_index=True,
    )
    output_df = output_df.rename(
        columns={
            p_col: a_col
            for p_col, a_col in zip(probability_columns, amount_columns)
        }
    )
    filters = []
    if forecast_start_month:
        filters.append(f"date >= '{forecast_start_month}'")
    if forecast_end_month:
        filters.append(f"date <= '{forecast_end_month}'")

    if len(filters) > 0:
        return output_df.query(" and ".join(filters))

    return output_df


def expand_rows(row, probability_columns):
    date_range = pd.date_range(
        start=row["modified_start_date"], end=row["end_date"], freq="MS"
    )
    res = {
        "record_id": row["record_id"],
        "date": date_range,
    }
    for prob_col in probability_columns:
        if row["expected_project_duration_in_months"] == 0:
            split_amount = 0.0
        else:
            split_amount = (
                    row[prob_col]
                    * (
                            row["amount_in_company_currency"]
                            + row["tcv_and_amount_delta"]
                    )
                    / row["expected_project_duration_in_months"]
            )
        res.update({prob_col: split_amount})
        # print(res)
    return pd.DataFrame(res)


def calculate_forecasts(
        df, probability_columns, amount_columns, forecast_end_month, rename=True
):
    _forecasts = [OrderedDict() for _ in probability_columns]
    unique_snapshot_dates = sorted(list(df["snapshot_month"].unique()))
    for date in unique_snapshot_dates:
        forecast_df = _calculate_forecasts(
            df[df.snapshot_month == date].copy(),
            probability_columns,
            amount_columns,
            date + pd.DateOffset(days=1),
            forecast_end_month,
        )
        for index, a_col in enumerate(amount_columns):
            if rename:
                _forecasts[index][date] = (
                    forecast_df[["record_id", "date", a_col]]
                    .rename(columns={a_col: "amount"})
                    .copy()
                )
            else:
                _forecasts[index][date] = forecast_df[
                    ["record_id", "date", a_col]
                ].copy()
    if len(amount_columns) == 1:
        return _forecasts[0]
    return _forecasts


def prepare_results(res):
    results_df = (sum([r.T for r in res])).copy()
    results_df = results_df.drop(
        results_df.isna().mean()[results_df.isna().mean() == 1].index, axis=1
    )
    return results_df.T


def _calculate_snapshot_monthly_number(date_string, experiment_name):
    # print(date_string, experiment_name)
    dates = [
        datetime.strptime(d.replace("'", ""), "%Y-%m-%d").date()
        for d in date_string.split(",")
    ]
    # print(st.session_state["forecast_results"])
    if "forecast_results" in st.session_state:
        experiment_forecasts = st.session_state["forecast_results"].get(
            experiment_name, OrderedDict()
        )
        if len(experiment_forecasts) > 0:
            _results = OrderedDict()
            for d in dates:
                forecast_df = experiment_forecasts[d].copy()
                # st.write(forecast_df)
                all_ids = st.session_state["hubspot_id_dict"][d]

                _act = st.session_state["actual_df"][
                    st.session_state["actual_df"].hubspot_id.isin(all_ids)
                    & (st.session_state["actual_df"].date >= d)
                    ].copy()

                mapping = {}
                for i, m in enumerate(forecast_df.date.drop_duplicates().sort_values()):
                    mapping[m] = f"M{i + 1}"
                forecast_df.date = forecast_df.date.map(mapping)
                _act.date = _act.date.map(mapping)
                result_df = pd.DataFrame()
                result_df["Forecast"] = forecast_df.groupby(
                    "date",
                ).amount.sum()
                result_df["Actual"] = _act.groupby("date").amount.sum()
                result_df["MAPE"] = 100.0 * abs(
                    1
                    - forecast_df.groupby("date").amount.sum()
                    / _act.groupby("date").amount.sum()
                )
                _results[d] = result_df.copy()

            return _results
    return {}


def get_ids_for_each_cohort(active_dff, forecast_dff):
    # cohort_df = cohort_df.merge(forecast_df, on="record_id", how="left")
    temp = forecast_dff.merge(active_dff, on="record_id", how="inner")
    cohort_id_dict = {}
    for cohort in temp.cohort.unique():
        cohort_id_dict[cohort] = temp[temp.cohort == cohort].record_id.unique()
    return cohort_id_dict
    # return cohort_df


def map_cohort_to_ids(dict_, df, id_col):
    id_to_cohort = {
        id_: cohort for cohort, ids in dict_.items() for id_ in ids
    }
    df["cohort"] = df[id_col].map(id_to_cohort)
    return df


def calculate_cohort_error(date_string, experiment_name):
    if "forecast_results" in st.session_state:
        current_forecasts = st.session_state["forecast_results"].get(
            experiment_name, OrderedDict()
        )
        existing_forecasts = st.session_state["forecast_results"].get(
            "Existing Approach", OrderedDict()
        )
        default_forecasts = st.session_state["forecast_results"].get(
            "Default", OrderedDict()
        )
        _results = []
        for d in current_forecasts.keys():
            # st.write(d)
            # st.write(current_forecasts[d])
            cohort_wise = st.session_state["active_df"][st.session_state["active_df"].snapshot_date.dt.date == d]
            [
                [
                    "record_id",
                    "cohort",
                    # *st.session_state["cohort_selected_features"],
                    "final_probability",
                ]
            ].copy()
            # st.write("Snapshot Date:", d)
            # st.write("Cohort Wise:", cohort_wise)
            # st.write("Current Forecasts:", current_forecasts[d])
            forecasts_data = (
                current_forecasts[d]
                .groupby("record_id")
                .amount.sum()
                .reset_index()
                .rename(columns={"amount": "current"})
            )
            # st.write("Forecasts Data before merging:", forecasts_data)
            # st.write(forecasts_data)
            forecasts_data = cohort_wise.merge(
                forecasts_data, on="record_id", how="left"
            )
            # st.write("Forecasts Data after merging:", forecasts_data)
            # st.write(forecasts_data)
            _act = (
                st.session_state["actual_df"][
                    st.session_state["actual_df"].hubspot_id.isin(
                        forecasts_data.record_id
                    )
                    & (st.session_state["actual_df"].date >= d)
                    ]
                .groupby("hubspot_id")
                .amount.sum()
                .reset_index()
                .rename(
                    columns={"hubspot_id": "record_id", "amount": "actual"}
                )
            )

            forecasts_data = forecasts_data.merge(
                _act, on="record_id", how="left"
            )
            # st.write("Before filling NA:", forecasts_data)
            forecasts_data = forecasts_data.fillna(0.0)
            forecasts_data = forecasts_data.merge(
                existing_forecasts[d]
                .groupby("record_id")
                .amount.sum()
                .reset_index()
                .rename(columns={"amount": "existing"}),
                on="record_id",
                how="left",
            ).fillna(0.0)
            forecasts_data = forecasts_data.merge(
                default_forecasts[d]
                .groupby("record_id")
                .amount.sum()
                .reset_index()
                .rename(columns={"amount": "default"}),
                on="record_id",
                how="left",
            ).fillna(0.0)
            # st.write("Forecasts Data after merging with Default & Existing:", forecasts_data)
            forecasts_data = forecasts_data.groupby(
                [
                    "cohort",
                    # *st.session_state["cohort_selected_features"],
                    # "final_probability",
                ]
            )[
                ["actual", "existing", "default", "current", 'final_probability']
            ].agg({"actual": "sum", "existing": "sum", "default": "sum", "current": "sum",
                   'final_probability': 'mean'}).reset_index()
            # st.write("Forecasts Data after grouping by cohort:", forecasts_data)
            forecasts_data = forecasts_data.set_index("cohort")
            _results.append(forecasts_data.copy())

        concatenated_df = pd.concat(_results)
        cohort_result_df = concatenated_df.groupby("cohort").mean().reset_index()
        cohort_result_df["error_current"] = 100 * abs(
            1 - cohort_result_df["current"] / cohort_result_df["actual"]
        )
        cohort_result_df["error_existing"] = 100 * abs(
            1 - cohort_result_df["existing"] / cohort_result_df["actual"]
        )
        cohort_result_df["error_default"] = 100 * abs(
            1 - cohort_result_df["default"] / cohort_result_df["actual"]
        )
        return cohort_result_df


def aggregate_snapshot_numbers(snapshot_numbers):
    concatenated = pd.concat(
        [res.reset_index() for res in snapshot_numbers.values()]
    )
    return concatenated.groupby("date")[["Forecast", "Actual", "MAPE"]].mean()


def prepare_bu_revenue_data():
    hist_actual_df = pd.read_excel(
        r"data/Blend Consolidated Historical Revenue 140824.xlsx"
    )
    # st.session_state['hist_actual_df'] = hist_actual_df.copy()
    return hist_actual_df


def preprocess_bu_rev(actual_df):
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

    df_list = np.split(actual_df, actual_df[actual_df.isnull().all(1)].index)
    df_list = list(filter(lambda x: x.dropna(how='all').shape[0] > 0, df_list))
    df_list = [df_.pipe(drop_null_rows_columns) \
                   .pipe(set_column_header) \
                   .pipe(convert_to_long_form) \
                   .pipe(create_date_column) \
                   .pipe(convert_output_to_floats)
               for df_ in df_list]
    map_bu = {
        0: 'BTS',
        1: 'DSX',
        2: 'EMEA',
        3: 'MLABS',
        4: 'FPAI'
    }
    for i, df_ in enumerate(df_list):
        df_list[i]['bu'] = map_bu[i]
    return pd.concat(df_list, ignore_index=True)

def create_exo_df():
    csv_url = f'https://raw.githubusercontent.com/rohankblend/Forecasting_exog_files/master/exog_variables.csv'
    exo_df = pd.read_csv(csv_url, index_col=0, )
    print(exo_df.DATE.head(20))
    exo_df.DATE = pd.to_datetime(exo_df.DATE, format = '%Y-%m-%d')

    for col in exo_df.columns:
        if exo_df[col].dtype == 'object':
            exo_df[col] = exo_df[col].str.replace(',', '').astype(np.float64)

    exo_df['ds'] = exo_df.DATE.apply(lambda x: datetime(x.year, x.month, 1))
    exo_df = exo_df.drop('DATE', axis=1).fillna(0).groupby('ds').mean().reset_index()
    df_ = [
        ['2024-07-01', 5500, 2400, 85, 3.1],
        ['2024-08-01', 5550, 2420, 88, 3.0],
        ['2024-09-01', 5600, 2440, 90, 2.9],
        ['2024-10-01', 5625, 2460, 92, 2.8],
        ['2024-11-01', 5650, 2480, 94, 2.8],
        ['2024-12-01', 5675, 2500, 96, 2.8]
    ]

    # Create a DataFrame
    df_ = pd.DataFrame(df_, columns=['month', 'S&P500', 'GOLD_PRICE', 'CRUDE_PRICE', 'CPI'])
    print(df_)
    df_.month = pd.to_datetime(df_.month)
    df_[['S&P500', 'GOLD_PRICE', 'CRUDE_PRICE', 'CPI']] = df_.drop('month', axis=1).astype('float')

    df_ = df_.rename(columns={'month': 'ds'})
    df_['CPI'] *= 100
    exo_df = pd.concat([exo_df[exo_df.ds <= '2024-06-01'], df_
                        ], axis=0)
    exo_df = exo_df.reset_index(drop=True)
    return exo_df
