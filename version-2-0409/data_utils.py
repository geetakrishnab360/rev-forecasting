import pandas as pd
from functools import reduce
import numpy as np
from collections import OrderedDict
import streamlit as st
from datetime import datetime


def prepare_actual_rev_data():
    actual_df = pd.read_excel(
        r"C:\Users\KarthikKurugodu\PycharmProjects\streamlit-tool\streamlit-tool\actuals.xlsx"
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


def convert_to_cohort_df(experiemnt_df):
    features = experiemnt_df["columns"].str.split(
        "|",
        expand=True,
        regex=False,
    )
    features.columns = features.iloc[0]
    features = features.iloc[1:]
    features["eff_probability"] = (
        experiemnt_df["probabilities"].iloc[1:].copy()
    )
    features = features.reset_index(drop=True)
    features["cohort"] = pd.Series(
        [f"cohort {i}" for i in range(1, len(experiemnt_df))]
    )
    # features.columns = features.iloc[0]
    # features.drop(0, inplace=True)
    return features


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
    df[string_columns] = df[string_columns].transform(lambda x: x.str.strip())
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


def preprocess(
        dataframe,
):
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


def _calculate_forecasts(df, forecast_end_month, forecast_start_month):
    _df = df.copy()
    _df["end_date"] = _df.apply(
        lambda x: x["modified_start_date"] + pd.DateOffset(months=int(x["expected_project_duration_in_months"]) - 1),
        axis=1)

    def expand_rows(row):
        date_range = pd.date_range(start=row["modified_start_date"], end=row["end_date"], freq="MS")
        if row["expected_project_duration_in_months"] == 0:
            split_amount = 0.0
        else:
            split_amount = (
                    row["final_probability"]
                    * (
                            row["amount_in_company_currency"]
                            + row["tcv_and_amount_delta"]
                    )
                    / row["expected_project_duration_in_months"]
            )
        return pd.DataFrame(
            {
                "record_id": row["record_id"],
                "date": date_range,
                "amount": split_amount,
            }
        )

    output_df = pd.concat(_df.apply(expand_rows, axis=1).tolist(), ignore_index=True)
    output_df = output_df[
        (output_df.date <= forecast_end_month)
        & (output_df.date >= forecast_start_month)
        ]
    return output_df


def calculate_forecasts(df, forecast_end_month):
    _forecasts = OrderedDict()
    unique_snapshot_dates = sorted(list(df["snapshot_month"].unique()))
    for date in unique_snapshot_dates:
        forecast_df = _calculate_forecasts(
            df[df.snapshot_month == date].copy(),
            forecast_end_month,
            date + pd.DateOffset(days=1),
        )
        _forecasts[date] = forecast_df.copy()

    return _forecasts


def prepare_results(res):
    results_df = (sum([r.T for r in res])).copy()
    results_df = results_df.drop(
        results_df.isna().mean()[results_df.isna().mean() == 1].index, axis=1
    )
    return results_df.T


# @st.cache_data()
def _calculate_snapshot_monthly_number(date_string, experiment_name):
    print(date_string, experiment_name)
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
            # st.write(experiment_forecasts.keys())
            for d in dates:
                # st.write(d)
                # st.write(st.session_state['active_df'])
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


def _calculate_cohort_error(date_string, experiment_name):
    pass


#     dates = [
#         datetime.strptime(d.replace("'", ""), "%Y-%m-%d").date()
#         for d in date_string.split(",")
#     ]
#
#     if "data_df" in st.session_state and "cohort_df" in st.session_state and len(st.session_state['cohort_df']) > 0:
#         experiment_forecasts = st.session_state["forecast_results"].get(
#             experiment_name, OrderedDict()
#         )
#         activedf = st.session_state["data_df"].copy()
#         active_df_merged_coh_df = activedf.merge(st.session_state['cohort_df'],
#                                                  on=st.session_state.get('cohort_selected_features', []), how='left')
#         st.write(active_df_merged_coh_df)

# active_df_merged_coh_df = active_df_merged_coh_df[['record_id', 'cohort', 'snapshot_month']]
# if len(experiment_forecasts) > 0:
#     _results = OrderedDict()
#     # st.write(experiment_forecasts.keys())
#     for d in dates:
#         # st.write(d)
#         forecast_df = experiment_forecasts[d].copy()
#         temp = active_df_merged_coh_df[active_df_merged_coh_df.snapshot_month == d]
#
#         # st.write(forecast_df)
#         all_ids = st.session_state["hubspot_id_dict"][d]
#
#         _act = st.session_state["actual_df"][
#             st.session_state["actual_df"].hubspot_id.isin(all_ids)
#             & (st.session_state["actual_df"].date >= d)
#             ].copy()
#
#         mapping = {}
#         for i, m in enumerate(forecast_df.date.drop_duplicates().sort_values()):
#             mapping[m] = f"M{i + 1}"
#         forecast_df.date = forecast_df.date.map(mapping)
#         _act.date = _act.date.map(mapping)
#         result_df = pd.DataFrame()
#         result_df["Forecast"] = forecast_df.groupby(
#             "date",
#         ).amount.sum()
#         result_df["Actual"] = _act.groupby("date").amount.sum()
#         result_df["MAPE"] = 100.0 * abs(
#             1
#             - forecast_df.groupby("date").amount.sum()
#             / _act.groupby("date").amount.sum()
#         )
#         _results[d] = result_df.copy()
# st.write(st.session_state['active_df'])


# def _calculate_cohort_error(date_string, experiment_name):
#     print(date_string, experiment_name)
#     dates = [
#         datetime.strptime(d.replace("'", ""), "%Y-%m-%d").date()
#         for d in date_string.split(",")
#     ]
#
#     # print(st.session_state["forecast_results"])
#     if "forecast_results" in st.session_state:
#         experiment_forecasts = st.session_state["forecast_results"].get(
#             experiment_name, OrderedDict()
#         )
#         if len(experiment_forecasts) > 0:
#             _results = OrderedDict()
#             for d in dates:
#                 forecast_df = experiment_forecasts[d].copy()
#                 actual_df = st.session_state["actual_df"].copy()
#                 st.write(forecast_df)
#                 # all_ids = st.session_state["hubspot_id_dict"][d]
#                 temp = actual_df.merge(st.session_state['cohort_df'], on=st.session_state['cohort_selected_features'],
#                                        how='inner')
#                 st.write
#                 st.write(temp)
#             _act = st.session_state["actual_df"][
#                 st.session_state["actual_df"].hubspot_id.isin(all_ids)
#                 & (st.session_state["actual_df"].date >= d)
#                 ].copy()
#
#             mapping = {}
#             for i, m in enumerate(forecast_df.date.drop_duplicates().sort_values()):
#                 mapping[m] = f"M{i + 1}"
#             forecast_df.date = forecast_df.date.map(mapping)
#             _act.date = _act.date.map(mapping)
#             result_df = pd.DataFrame()
#             result_df["Forecast"] = forecast_df.groupby(
#                 "date",
#             ).amount.sum()
#             result_df["Actual"] = _act.groupby("date").amount.sum()
#             result_df["MAPE"] = 100.0 * abs(
#                 1
#                 - forecast_df.groupby("date").amount.sum()
#                 / _act.groupby("date").amount.sum()
#             )
#             _results[d] = result_df.copy()
#
#         return _results
# return {}


def aggregate_snapshot_numbers(snapshot_numbers):
    concatenated = pd.concat(
        [res.reset_index() for res in snapshot_numbers.values()]
    )
    return concatenated.groupby("date")[["Forecast", "Actual", "MAPE"]].mean()
