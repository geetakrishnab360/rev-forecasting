import streamlit as st
import plotly.graph_objs as go
from io import BytesIO
from plotly.subplots import make_subplots
from plotly import colors
from dotenv import load_dotenv
from scipy.optimize import minimize
from snowflake_utils import (
    convert_period_to_dates,
    convert_dates_to_string,
    fetch_data_from_db, fetch_weightages,
)
import pandas as pd
from data_utils import (
    preprocess,
    calculate_forecasts,
    _calculate_snapshot_monthly_number,
    calculate_cohort_error,
    convert_to_cohort_df,
    prepare_actual_rev_data,
    aggregate_snapshot_numbers,
)

from db import (
    create_database,
    insert_experiment_into_db,
    fetch_all_experiments,
    fetch_experiment,
)

from components import set_header
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# from db import fetch_experiment
# from data_utils import convert_to_cohort_df

st.set_page_config(layout="wide", page_title="Blend Forecasting", initial_sidebar_state="collapsed")
load_dotenv()
prepare_actual_rev_data()
set_header("Forecasting Simulation - Pipeline Analysis")
create_database()


def enable_save_experiment():
    st.session_state["disable_save_experiment"] = (
            len(st.session_state["experiment"].strip()) == 0
            or len(st.session_state["cohort_df"]) == 0
    )


def disable_calculations():
    st.session_state["disable_calculations"] = True


def format_period_text(period):
    if period > 1:
        return f"{period} months"
    else:
        return f"{period} month"


def fetch_and_forecast_experiment(experiment_name):
    exp_df = convert_to_cohort_df(fetch_experiment("karthik", experiment_name))
    exp_df['selected'] = exp_df.selected.map({1: True, 0: False})
    cohort_selected_features = list(
        exp_df.drop(["cohort", "eff_probability", "selected"], axis=1).columns
    )

    st.session_state["cohort_information"][experiment_name] = {}
    st.session_state["cohort_information"][experiment_name][
        "cohort_df"
    ] = exp_df.copy()
    st.session_state["cohort_information"][experiment_name][
        "cohort_selected_features"
    ] = cohort_selected_features.copy()

    # st.write(st.session_state["cohort_df"])
    _df = st.session_state["data_df"].copy()
    _df[cohort_selected_features] = _df[cohort_selected_features].fillna(
        "None"
    )

    _df = _df.merge(exp_df, on=cohort_selected_features, how="left").copy()

    _df["final_probability"] = _df["eff_probability"].fillna(0)
    temp = _df.copy()

    if not exp_df.iloc[0]['selected']:
        ids = temp[
            temp['deal_probability'] != temp['effective_probability']
            ].index
        temp.loc[ids, 'final_probability'] = temp.loc[
            ids, 'effective_probability']

        temp.loc[ids, 'cohort'] = 'cohort 0'

    _df = temp.copy()
    forecast_results = calculate_forecasts(
        _df, ["final_probability"], ["amount"], "2024-07-01"
    )

    st.session_state["forecast_results"][
        experiment_name
    ] = forecast_results.copy()
    return exp_df, cohort_selected_features, _df


def change_experiment():
    st.session_state["selected_experiment_name"] = st.session_state[
        "selected-experiment"
    ]

    if st.session_state["selected_experiment_name"] != "Existing Approach":
        (
            st.session_state["cohort_df"],
            st.session_state["cohort_selected_features"],
            st.session_state["active_df"],
        ) = fetch_and_forecast_experiment(
            st.session_state["selected_experiment_name"]
        )


def pull_data():
    if st.session_state.get("enter-custom-dates", False):
        pass
    else:
        start_date, end_date = convert_period_to_dates(
            st.session_state.get("data_period", 6)
        )
        st.session_state["dates_string"] = convert_dates_to_string(start_date, end_date)

    data = fetch_data_from_db(st.session_state["dates_string"])

    st.session_state["hubspot_id_dict"], st.session_state["data_df"] = (
        preprocess(data)
    )

    if "forecast_results" in st.session_state:
        del st.session_state["forecast_results"]


def cohort_generate_forecast(experiment_name="Current"):
    # st.write(st.session_state["cohort_df"])
    st.session_state["cohort_information"]["Current"] = {}
    # st.session_state["cohort_information"]["Current"]["cohort_df"] = (
    #     st.session_state["cohort_df"].copy()
    # )
    st.session_state["cohort_df"] = edited_data.copy()
    st.session_state["cohort_information"]["Current"]["cohort_df"] = (
        st.session_state["cohort_df"].copy()
    )
    st.session_state["cohort_information"]["Current"]['cohort_selected_features'] = st.session_state[
        "cohort_selected_features"].copy()
    st.session_state["active_df"] = st.session_state["data_df"].copy()
    st.session_state["active_df"][
        st.session_state.get("cohort_selected_features", [])
    ] = st.session_state["active_df"][
        st.session_state.get("cohort_selected_features", [])
    ].fillna(
        "None"
    )
    st.session_state["active_df"] = (st.session_state["active_df"].merge
        (
        edited_data,
        on=st.session_state.get("cohort_selected_features", []),
        how="left",
    )).copy()

    # st.write(st.session_state["active_df"])
    st.session_state["active_df"]["final_probability"] = st.session_state[
        "active_df"
    ]["eff_probability"].fillna(0)
    temp = st.session_state['active_df'].copy()

    if not edited_data.iloc[0]['selected']:
        ids = temp[
            temp['deal_probability'] != temp['effective_probability']
            ].index
        temp.loc[ids, 'final_probability'] = temp.loc[
            ids, 'effective_probability']

        temp.loc[ids, 'cohort'] = 'cohort 0'

    # ids = temp[
    #     temp['deal_probability'] != temp['effective_probability']
    #     ].index
    # temp.loc[ids, 'final_probability'] = temp.loc[
    #     ids, 'effective_probability']
    #
    # temp.loc[ids, 'cohort'] = 'cohort 0'

    st.session_state["active_df"] = temp.copy()
    forecast_results = calculate_forecasts(
        st.session_state["active_df"],
        ["final_probability"],
        ["amount"],
        "2024-07-01",
    )
    # for key, value in forecast_results.items():
    #     st.write(key)
    #     st.write(value)
    st.session_state["forecast_results"]["Current"] = forecast_results.copy()
    st.session_state["update_probabilities_clicked"] = True


if "update_probabilities_clicked" not in st.session_state:
    st.session_state["update_probabilities_clicked"] = False


def save_experiment_to_db():
    insert_experiment_into_db(
        st.session_state["cohort_df"],
        st.session_state["cohort_selected_features"],
        st.session_state["experiment"],
        "karthik",
    )

    st.session_state["all_experiments"] = fetch_all_experiments("karthik")


# def cohorts_update(selected_features):
#     if len(selected_features) == 0:
#         st.session_state["cohort_df"] = pd.DataFrame()
#         return
#     cohort_df = (
#         st.session_state["data_df"][selected_features]
#         .drop_duplicates()
#         .reset_index(drop=True)
#     )
#
#     num_cohorts = cohort_df.shape[0]
#     cohort_df["cohort"] = [f"cohort {i}" for i in range(1, num_cohorts + 1)]
#     # st.write("Before adding prob", cohort_df)
#     if "deal_stage" in selected_features:
#         deal_stage_and_probability_dict = (
#             st.session_state["data_df"]
#             .set_index("deal_stage")["deal_probability"]
#             .to_dict()
#         )
#         cohort_df["eff_probability"] = cohort_df["deal_stage"].map(
#             deal_stage_and_probability_dict
#         )
#     else:
#         selected_features_and_probability = st.session_state["data_df"][
#             selected_features + ["final_probability"]
#             ]
#         t = (
#             selected_features_and_probability.groupby(
#                 selected_features, dropna=False
#             )
#             .final_probability.mean()
#             .reset_index()
#         )
#         cohort_df = cohort_df.merge(t, on=selected_features, how="left")
#         cohort_df.rename(
#             columns={"final_probability": "eff_probability"}, inplace=True
#         )
#
#     st.session_state["cohort_df"] = cohort_df[
#         ["cohort", *selected_features, "eff_probability"]
#     ]
#     st.session_state["cohort_df"] = st.session_state["cohort_df"].fillna(
#         "None"
#     )
#     st.session_state["disable_calculations"] = False
#     st.session_state["cohort_information"]["Current"] = {}
#     st.session_state["cohort_information"]["Current"]["cohort_df"] = (
#         st.session_state["cohort_df"].copy()
#     )
#     st.session_state["cohort_information"]["Current"][
#         "cohort_selected_features"
#     ] = selected_features.copy()
#     enable_save_experiment()

def cohorts_update(selected_features):
    if len(selected_features) == 0:
        st.session_state["cohort_df"] = pd.DataFrame()
        return

    # Create a cohort for rows where deal_probability is not equal to effective_probability
    data_df = st.session_state["data_df"].copy()
    cohort_0 = data_df[data_df["deal_probability"] != data_df["effective_probability"]]
    # print(len(cohort_0))
    cohort_0["cohort"] = "cohort 0"

    # Remove these rows from the main DataFrame
    data_df = data_df[data_df["deal_probability"] == data_df["effective_probability"]]

    # Create cohorts based on selected features
    cohort_df = data_df[selected_features].drop_duplicates().reset_index(drop=True)
    num_cohorts = cohort_df.shape[0]
    cohort_df["cohort"] = [f"cohort {i + 1}" for i in range(num_cohorts)]

    if "deal_stage" in selected_features:
        deal_stage_and_probability_dict = (
            st.session_state["weights_df"]
            .set_index("deal_stage")["weightage"]
            .to_dict()
        )
        # print("Deal stage and probability dict:", deal_stage_and_probability_dict)
        cohort_df["eff_probability"] = cohort_df["deal_stage"].map(
            deal_stage_and_probability_dict
        )
    else:
        selected_features_and_probability = st.session_state["data_df"][
            selected_features + ["final_probability"]
            ]
        t = (
            selected_features_and_probability.groupby(
                selected_features, dropna=False
            )
            .final_probability.mean()
            .reset_index()
        )
        cohort_df = cohort_df.merge(t, on=selected_features, how="left")
        cohort_df.rename(
            columns={"final_probability": "eff_probability"}, inplace=True
        )
    c0 = cohort_0[['effective_probability', 'cohort']].copy()
    c0['eff_probability'] = c0['effective_probability'].groupby(c0['cohort']).transform('mean')
    c0 = c0.drop_duplicates(subset=['cohort'])
    c0[selected_features] = '-'
    # Combine cohort 0 with other cohorts
    cohort_df = pd.concat([c0, cohort_df], ignore_index=True)

    st.session_state["cohort_df"] = cohort_df[
        ["cohort", *selected_features, "eff_probability"]
    ]
    st.session_state["cohort_df"] = st.session_state["cohort_df"].fillna(
        "None"
    )
    st.session_state["disable_calculations"] = False
    # st.session_state["cohort_information"]["Current"] = {}
    # st.session_state["cohort_information"]["Current"]["cohort_df"] = (
    #     st.session_state["cohort_df"].copy()
    # )
    # st.session_state["cohort_information"]["Current"][
    #     "cohort_selected_features"
    # ] = selected_features.copy()
    enable_save_experiment()


def allow_calculation():
    st.session_state["disable_calculations"] = False


def calculate_current_vs_act_error(eff_probabilities, temp, cohorts, actuals):
    selected_cohort_df = st.session_state["cohort_df"][st.session_state["cohort_df"]["selected"]].copy()
    selected_cohort_df = selected_cohort_df[(selected_cohort_df["cohort"] != 'cohort 0')].copy()
    selected_cohort_df["eff_probability"] = eff_probabilities
    prob_map = dict(zip(cohorts, eff_probabilities))
    temp["final_probability"] = temp.cohort.map(prob_map).fillna(0.0)

    forecast_results = calculate_forecasts(temp, ["final_probability"], ["amount"], "2024-07-01")
    st.session_state["forecast_results"]["Current"] = forecast_results.copy()

    total_forecasts = np.array([_res["amount"].sum().item() for _res in forecast_results.values()])
    return np.mean(abs(1 - total_forecasts / actuals))


def optimize_eff_probability(max_iter, progress_bar, progress_text):
    st.session_state["cohort_df"] = edited_data.copy()
    st.session_state["active_df"] = st.session_state["data_df"].copy()
    st.session_state["active_df"][
        st.session_state.get("cohort_selected_features", [])
    ] = st.session_state["active_df"][
        st.session_state.get("cohort_selected_features", [])
    ].fillna(
        "None"
    )
    st.session_state["active_df"] = (
        st.session_state["active_df"]
        .merge(
            edited_data,
            on=st.session_state.get("cohort_selected_features", []),
            how="left",
        )
        .copy()
    )
    # st.write(st.session_state["active_df"])
    st.session_state["active_df"]["final_probability"] = st.session_state[
        "active_df"
    ]["eff_probability"].fillna(0)
    temp = st.session_state['active_df'].copy()
    # ids = temp[
    #     temp['deal_probability'] != temp['effective_probability']
    #     ].index
    # temp.loc[ids, 'final_probability'] = temp.loc[
    #     ids, 'effective_probability']
    #
    # temp.loc[ids, 'cohort'] = 'cohort 0'
    if not edited_data.iloc[0]['selected']:
        ids = temp[temp['deal_probability'] != temp['effective_probability']].index
        temp.loc[ids, 'final_probability'] = temp.loc[ids, 'effective_probability']
        temp.loc[ids, 'cohort'] = 'cohort 0'
    temp1 = temp.copy()

    _act_df = st.session_state["active_df"].merge(
        st.session_state["actual_df"], left_on="record_id", right_on="hubspot_id"
    )
    _act_df["date"] = pd.to_datetime(_act_df["date"])
    _act_df["month_diff"] = (_act_df["date"].dt.year - _act_df["snapshot_date"].dt.year) * 12 + (
            _act_df["date"].dt.month - _act_df["snapshot_date"].dt.month)
    _act_df = _act_df[_act_df["month_diff"] >= 1].copy()
    _actuals = _act_df.groupby("snapshot_date").amount.sum().values

    selected_cohort_df_1 = st.session_state["cohort_df"][
        (st.session_state["cohort_df"]["selected"] == True) & (
            (st.session_state["cohort_df"]["cohort"] != 'cohort 0')
        )
        ].copy()
    initial_guess = selected_cohort_df_1["eff_probability"].values
    st.session_state["initial_probabilities"] = initial_guess.copy()
    cohorts = selected_cohort_df_1.cohort.tolist()
    bounds = [(0, 1) for _ in range(len(initial_guess))]

    def progress_callback():
        progress = min(1.0, progress_callback.iteration / max_iter)
        progress_bar.progress(progress)
        progress_text.text(f"Optimization progress: {int(progress * 100)}%")
        progress_callback.iteration += 1

    progress_callback.iteration = 0

    result = minimize(
        calculate_current_vs_act_error,
        initial_guess,
        bounds=bounds,
        method="SLSQP",
        options={"maxiter": max_iter, "ftol": 1e-4, "disp": False},
        callback=progress_callback,
        args=(temp, cohorts, _actuals),
    )

    if result.success:
        optimized_eff_probabilities = result.x
        st.session_state["cohort_df"].loc[
            st.session_state["cohort_df"]["cohort"].isin(selected_cohort_df_1["cohort"]), "eff_probability"
        ] = optimized_eff_probabilities
        # st.session_state["cohort_df"].loc[
        #     selected_cohort_df_1['cohort'], "eff_probability"] = optimized_eff_probabilities
        prob_map = dict(zip(selected_cohort_df_1.cohort, optimized_eff_probabilities))
        temp["final_probability"] = temp.cohort.map(prob_map).fillna(temp1["final_probability"])

        # if edited_data.iloc[0]['selected']:
        #     ids = temp[temp['deal_probability'] != temp['effective_probability']].index
        #     temp.loc[ids, 'final_probability'] = st.session_state["cohort_df"].loc[
        #         st.session_state["cohort_df"]['cohort'] == 'cohort 0', "eff_probability"].values[0]
        #     temp.loc[ids, 'final_probability'] = temp.loc[ids, 'final_probability']
        #     temp.loc[ids, 'cohort'] = 'cohort 0'

        st.session_state["active_df"] = temp.copy()
        st.session_state["optimization_done"] = True
        st.session_state["optimization_successful"] = True
        forecast_results = calculate_forecasts(
            st.session_state["active_df"],
            ["final_probability"],
            ["amount"],
            "2024-07-01",
        )
        st.session_state["forecast_results"]["Current"] = forecast_results.copy()
        st.session_state['cohort_information']['Current']['cohort_df'] = st.session_state["cohort_df"].copy()
        # print(forecast_results)
        st.rerun()
        # st.write(forecast_results)
    else:
        st.error("Optimization failed")

    if st.session_state.get("optimization_successful"):
        st.success("Optimization successful!")
        st.session_state["optimization_successful"] = False

    progress_bar.progress(1.0)
    progress_text.text("Optimization complete")


def reset_probabilities():
    if "initial_probabilities" in st.session_state:
        selected_cohort_df_1 = st.session_state["cohort_df"][
            (st.session_state["cohort_df"]["selected"] == True) & (
                (st.session_state["cohort_df"]["cohort"] != 'cohort 0')
            )
            ].copy()
        st.session_state["cohort_df"].loc[
            st.session_state["cohort_df"]["cohort"].isin(selected_cohort_df_1["cohort"]), "eff_probability"
        ] = st.session_state["initial_probabilities"]
        st.session_state["disable_calculations"] = False
        st.session_state["update_probabilities_clicked"] = False


with open("./styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )

if "cohort_df" not in st.session_state:
    st.session_state["cohort_df"] = pd.DataFrame()

if "data_df" not in st.session_state:
    pull_data()

if 'weights_df' not in st.session_state:
    st.session_state["weights_df"] = fetch_weightages()
    st.session_state["weights_df"].columns = [col.lower() for col in st.session_state["weights_df"].columns]
    st.session_state["weights_df"]["weightage"] = st.session_state["weights_df"]["weightage"].astype(float)

if "optimization_done" not in st.session_state:
    st.session_state["optimization_done"] = False

if "selected" not in st.session_state["cohort_df"]:
    # st.write("In IF loop")
    st.session_state["cohort_df"]["selected"] = True

if "future_df" not in st.session_state:
    data = fetch_data_from_db("'2024-07-31'")
    _, st.session_state["future_df"] = preprocess(data)

if "reporting-experiments" not in st.session_state:
    st.session_state["reporting-experiments"] = [
        "Existing Approach",
        "Default",
    ]

if "all_experiments" not in st.session_state:
    st.session_state["all_experiments"] = fetch_all_experiments("karthik")

if "forecast_results" not in st.session_state:
    st.session_state["forecast_results"] = {}

    forecast_results = calculate_forecasts(
        st.session_state["data_df"],
        ["final_probability", "deal_probability"],
        ["amount_existing", "amount_default"],
        "2024-07-01",
    )

    st.session_state["forecast_results"]["Existing Approach"] = (
        forecast_results[0].copy()
    )

    st.session_state["forecast_results"]["Default"] = forecast_results[
        1
    ].copy()
    # default_experiment()

if "cohort_information" not in st.session_state:
    st.session_state["cohort_information"] = {}

if "selected_experiment_name" not in st.session_state:
    st.session_state["selected_experiment_name"] = "Existing Approach"

if "final_all_sum_df" not in st.session_state:
    st.session_state["final_all_sum_df"] = pd.DataFrame()

left_pane, main_pane = st.columns((0.25, 0.75))

with left_pane:
    # Dropdown for period selection
    # if not st.session_state.get("enter-custom-dates", False):
    #     st.selectbox(
    #         "Data Period",
    #         options=[1, 3, 6],
    #         key="data_period",
    #         format_func=format_period_text,
    #         index=2,
    #     )
    # else:
    #     st.text_input(
    #         "Data Period",
    #         placeholder="type in simple english",
    #         key="data_period_text",
    #     )
    #
    # st.checkbox(
    #     "Enter Custom Dates",
    #     key="enter-custom-dates",
    # )

    # st.markdown(
    #     "<div class='date-period-text'>Selected Period : Jun 2024</div>",
    #     unsafe_allow_html=True,
    # )

    st.selectbox(
        "Data Period",
        options=[1, 3, 6],
        key="data_period",
        format_func=format_period_text,
        index=2,
    )

    st.button("Fetch Data", on_click=pull_data)

    # st.divider()

    st.text_input(
        "Save Experiment",
        key="experiment",
        on_change=enable_save_experiment,
    )

    _columns = st.columns(2)
    with _columns[0]:
        st.button(
            "Save Experiment",
            on_click=save_experiment_to_db,
            disabled=st.session_state.get("disable_save_experiment", True),
        )

    st.divider()

    st.selectbox(
        "Load Experiment",
        options=st.session_state["all_experiments"],
        key="selected-experiment",
        on_change=change_experiment,
        index=None,
    )

    st.multiselect(
        "Select Experiments for Comparison",
        options=["Existing Approach", "Default"]
                + st.session_state["all_experiments"],
        key="reporting-experiments",
    )

    # st.divider()

    # st.selectbox(
    #     "Select Period",
    #     options=["Quarter", "6 Months", "1 Year"],
    #     key="period",
    #     index=1
    # )
    # st.session_state['period_to_date'] = {"Quarter": "2024-10-01", "6 Months": "2025-01-01", "1 Year": "2025-07-01"}
    #
    # st.selectbox(
    #     "Select Experiment for report",
    #     options=['Current', 'Existing Approach', 'Default'] + st.session_state["all_experiments"],
    #     key='selected_report_experiment',
    #     index=None,
    # )
    #
    # st.text_input('Enter Excel file name (e.g. email_data.xlsx)', key='filename')

with main_pane:
    st.subheader("Cohort Creation")

    st.multiselect(
        "Features for cohorts",
        options=[
            "deal_stage",
            "engagement_type",
            "fulfillment_type",
            "pipeline",
            "deal_type",
        ],
        key="cohort_selected_features",
        on_change=disable_calculations,
    )

    st.button(
        "Update cohorts",
        on_click=cohorts_update,
        args=(st.session_state["cohort_selected_features"],),
        disabled=(
            True
            if len(st.session_state["cohort_selected_features"]) == 0
            else False
        ),
    )

    if len(st.session_state["cohort_df"]) > 0:
        column_config = {
            "cohort": st.column_config.TextColumn(
                "Cohort",
                disabled=True,
            ),
            "eff_probability": st.column_config.NumberColumn(
                "Effective Probabiity",
                disabled=False,
                required=True,
                max_value=1.0,
                step=0.01,
            ),
        }

        column_config.update(
            {
                f: st.column_config.TextColumn(
                    f.upper(),
                    disabled=True,
                )
                for f in st.session_state["cohort_selected_features"]
            }
        )

        st.text("Cohort data:")
        edited_data = st.data_editor(
            st.session_state["cohort_df"],
            key="edited_data",
            hide_index=True,
            column_config=column_config,
            on_change=allow_calculation,
        )

        st.button(
            "Update probabilities",
            on_click=cohort_generate_forecast,
            disabled=st.session_state.get("disable_calculations", False),
            type="primary",
        )

        # if 'active_df' in st.session_state:
        #     st.write(st.session_state['active_df'])
        #     # st.write(st.session_state['active_df'][
        #     #              st.session_state['active_df']['deal_probability'] != st.session_state['active_df'][
        #     #                  'final_probability']])

        max_iter = st.number_input(
            "Max Iterations",
            min_value=1,
            value=20,
            step=1,
            key="max_iter",
            label_visibility="collapsed",
        )
        if st.button(
                "Optimize Probabilities",
                disabled=not st.session_state.get(
                    "update_probabilities_clicked", False
                ),
                type="secondary",
        ):
            progress_bar = st.progress(0)
            progress_text = st.empty()
            optimize_eff_probability(max_iter, progress_bar, progress_text)

        if st.session_state.get("optimization_successful"):
            st.success("Optimization successful!")
            st.session_state["optimization_successful"] = False

        # st.write(st.session_state['active_df'])
        # st.write(st.session_state["cohort_df"])
        st.button(
            "Reset Probabilities",
            on_click=reset_probabilities,
            type="secondary",
            disabled=not st.session_state.get("optimization_done", False),  # Enable only after optimization
        )

    st.divider()
    with st.container():

        st.subheader("Results")
        agg_df = pd.DataFrame()
        column_order = []
        for expriment_name in ["Current"] + list(
                st.session_state["reporting-experiments"]
        ):
            if expriment_name not in st.session_state["forecast_results"]:
                if expriment_name == "Current":
                    continue
                else:
                    _ = fetch_and_forecast_experiment(expriment_name)

            snapshot_numbers = _calculate_snapshot_monthly_number(
                st.session_state["dates_string"], expriment_name
            )
            if len(agg_df) == 0:
                agg_df = (
                    aggregate_snapshot_numbers(snapshot_numbers)
                    .rename(
                        columns={
                            "Forecast": expriment_name,
                            "MAPE": f"{expriment_name} Vs Act (%Error)",
                            "Actual": "Actual",
                        }
                    )
                    .copy()
                )

            else:
                agg_df = agg_df.merge(
                    aggregate_snapshot_numbers(snapshot_numbers)
                    .drop("Actual", axis=1)
                    .rename(
                        columns={
                            "Forecast": expriment_name,
                            "MAPE": f"{expriment_name} Vs Act (%Error)",
                        }
                    )
                    .copy(),
                    left_index=True,
                    right_index=True,
                )

            column_order.extend(
                [expriment_name, f"{expriment_name} Vs Act (%Error)"]
            )

        if len(agg_df) == 0:
            st.write("No data available")
            st.stop()
        agg_df = agg_df[["Actual"] + column_order]

        # Creating subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Line Plot: Actual vs Existing Approach",
                "Bar Plot: MAPE for Each Month",
            ),
            horizontal_spacing=0.15,
        )

        # Add line plot (Actual and Existing Approach)
        fig.add_trace(
            go.Scatter(
                x=agg_df.index,
                y=agg_df["Actual"],
                mode="lines+markers",
                name="Actual",
            ),
            row=1,
            col=1,
        )
        for i, experiment_name in enumerate(
                ["Current"] + list(st.session_state["reporting-experiments"])
        ):
            if experiment_name not in st.session_state["forecast_results"]:
                if experiment_name == "Current":
                    continue

            # st.session_state[
            #     "reporting-experiments"
            # ]:
            fig.add_trace(
                go.Scatter(
                    x=agg_df.index,
                    y=agg_df[experiment_name],
                    mode="lines+markers",
                    name=experiment_name,
                    legendgroup=experiment_name,
                    line={"color": colors.DEFAULT_PLOTLY_COLORS[i + 1 % 10]},
                ),
                row=1,
                col=1,
            )

            # Add bar plot for MAPE (%Error)
            fig.add_trace(
                go.Bar(
                    x=agg_df.index,
                    y=agg_df[f"{experiment_name} Vs Act (%Error)"],
                    name=f"{experiment_name} MAPE",
                    legendgroup=experiment_name,
                    showlegend=False,
                    marker={"color": colors.DEFAULT_PLOTLY_COLORS[i + 1 % 10]},
                ),
                row=1,
                col=2,
            )

        # Update layout
        fig.update_layout(
            height=500,
            width=800,
            title_text="Actual vs Existing Approach and MAPE Subplots",
            barmode="group",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,  # Adjust the vertical position of the legend
                xanchor="center",
                x=0.5,
            ),
        )

        # Show the plot
        st.plotly_chart(fig, use_container_width=True)

        st.table(
            agg_df.T.style.set_table_styles(
                [
                    {
                        "selector": "thead  th",
                        "props": "background-color: #2d7dce; text-align:center; color: white;font-size:0.9rem;border-bottom: 1px solid black !important;",
                    },
                    {
                        "selector": "tbody  th",
                        "props": "font-weight:bold;font-size:0.9rem;color:#000;",
                    },
                    {
                        "selector": "tbody  tr:nth-child(2n + 3)",
                        "props": "background-color: #aacbec;",
                    },
                    {
                        "selector": "tbody  tr:nth-child(2n + 3) > td,tbody  tr:nth-child(2n + 3) > th",
                        "props": "border-bottom: 1px solid black !important;",
                    },
                    {
                        "selector": "tbody  tr > th",
                        "props": "border-right: 1px solid black !important;",
                    },
                ],
                overwrite=False,
            ).format(precision=0, thousands=",")
        )

        with st.expander("Snapshot Results"):

            st.markdown(
                f"<div class='selected-snapshot'>Selected Experiment : Current",
                unsafe_allow_html=True,
            )

            snapshot_numbers = _calculate_snapshot_monthly_number(
                st.session_state["dates_string"],
                "Current",
            )

            for date, res in snapshot_numbers.items():
                res = res.T
                ren_cols = []
                for i, cols in enumerate(res.columns):
                    ren_cols.append(
                        (date + pd.DateOffset(months=i + 1)).strftime("%b")
                    )
                res.columns = ren_cols
                snapshot_numbers[date] = res.T

            for date, res in snapshot_numbers.items():
                st.markdown(
                    f"<div class='date-period-text'>Snapshot : {date}</div>",
                    unsafe_allow_html=True,
                )
                st.table(
                    res.T.style.set_table_styles(
                        [
                            {
                                "selector": "thead  th",
                                "props": "background-color: #2d7dce; text-align:center; color: white;font-size:0.9rem;border-bottom: 1px solid black !important;",
                            },
                            {
                                "selector": "tbody  th",
                                "props": "font-weight:bold;font-size:0.9rem;color:#000;",
                            },
                            {
                                "selector": "tbody  tr:nth-child(2n + 3)",
                                "props": "background-color: #aacbec;",
                            },
                            {
                                "selector": "tbody  tr:nth-child(2n + 3) > td,tbody  tr:nth-child(2n + 3) > th",
                                "props": "border-bottom: 1px solid black !important;",
                            },
                            {
                                "selector": "tbody  tr > th",
                                "props": "border-right: 1px solid black !important;",
                            },
                        ],
                        overwrite=False,
                    ).format(precision=0, thousands=",")
                )

        # if 'active_df' in st.session_state:
        #     st.write(st.session_state['active_df'])

        with st.expander("Cohort Results"):
            st.markdown(
                "<div class='selected-snapshot'>Selected Experiment : Current</div>",
                unsafe_allow_html=True,
            )

            if "Current" in st.session_state["forecast_results"]:
                # st.write(st.session_state["forecast_results"]["Current"])
                coh_res = calculate_cohort_error(
                    st.session_state["dates_string"],
                    "Current",
                )
                # st.write(coh_res)
                coh_res = coh_res.rename(
                    columns={
                        "final_probability": "Probability",
                        "actual": "Actual",
                        "existing": "Existing",
                        "default": "Default",
                        "current": "Current",
                        "error_current": "Error (Current)",
                        "error_existing": "Error (Existing)",
                        "error_default": "Error (Default)",
                        "cohort": "Cohort",
                    }
                )
                coh_res = coh_res[
                    [
                        "Cohort",
                        # *st.session_state["cohort_selected_features"],
                        "Probability",
                        "Actual",
                        "Existing",
                        "Error (Existing)",
                        "Default",
                        "Error (Default)",
                        "Current",
                        "Error (Current)",
                    ]
                ]
                if coh_res is not None:
                    # st.write(coh_res)
                    st.table(
                        coh_res.set_index("Cohort")
                        .style.set_table_styles(
                            [
                                {
                                    "selector": "thead  th",
                                    # fmt: off
                                    "props": "background-color: #2d7dce; text-align:center; color: white;font-size:0.9rem;border-bottom: 1px solid black !important;",
                                    # fmt: on
                                },
                                {
                                    "selector": "tbody  th",
                                    "props": "font-weight:bold;font-size:0.9rem;color:#000;",
                                },
                                {
                                    "selector": "tbody  tr:nth-child(2n + 2)",
                                    "props": "background-color: #aacbec;",
                                },
                            ],
                            overwrite=False,
                        )
                        .format(precision=2)
                    )
                    # st.write(st.session_state['cohort_df'])
                else:
                    st.write("No Cohort Data available")

        # with st.expander("Future Forecasts"):
        #     if "sl_df" not in st.session_state:
        #         st.session_state["sl_df"] = pd.read_csv("sl_proportions.csv")
        #         st.session_state["sl_df"] = st.session_state["sl_df"][
        #             ~st.session_state["sl_df"].ASSOCIATED_DEAL_IDS.isna()].copy()
        #         st.session_state["sl_df"]["ASSOCIATED_DEAL_IDS"] = st.session_state["sl_df"][
        #             "ASSOCIATED_DEAL_IDS"].apply(lambda x: str(x).replace(".0", ""))
        #         st.session_state["sl_list"] = sorted(st.session_state["sl_df"].NAME.unique())
        #
        #     st.selectbox("Service line", options=["Overall"] + st.session_state["sl_list"], key="sl_selected", index=0)
        #
        #     st.session_state["future_df_"] = st.session_state["future_df"].copy()
        #     _exp_columns = []
        #
        #     for experiment_name in ["Current"] + st.session_state["reporting-experiments"]:
        #         cohort_info = st.session_state["cohort_information"].get(experiment_name, {})
        #         if cohort_info:
        #             cohort_df = cohort_info["cohort_df"]
        #             cohort_selected_features = cohort_info["cohort_selected_features"]
        #             st.session_state["future_df_"][cohort_selected_features] = st.session_state["future_df_"][
        #                 cohort_selected_features].fillna("None")
        #             st.session_state["future_df_"] = st.session_state["future_df_"].merge(
        #                 cohort_df[cohort_selected_features + ["eff_probability"]],
        #                 on=cohort_selected_features,
        #                 how="left"
        #             ).rename(columns={"eff_probability": f"{experiment_name}_prob"})
        #
        #             if not cohort_df.iloc[0]["selected"]:
        #                 ids = st.session_state["future_df_"][
        #                     st.session_state["future_df_"]['deal_probability'] != st.session_state["future_df_"][
        #                         'effective_probability']].index
        #                 st.session_state["future_df_"].loc[ids, f"{experiment_name}_prob"] = \
        #                     st.session_state["future_df_"].loc[ids, 'effective_probability']
        #
        #             _exp_columns.append(experiment_name)
        #
        #         if experiment_name == "Current":
        #             if len(st.session_state["forecast_results"].get("Current", {})) == 0:
        #                 if experiment_name in _exp_columns:
        #                     _exp_columns.remove(experiment_name)
        #                 continue
        #
        #     forecast_results = calculate_forecasts(
        #         st.session_state["future_df_"],
        #         [
        #             "final_probability",
        #             "deal_probability",
        #             *[f"{e}_prob" for e in _exp_columns],
        #         ],
        #         [
        #             "amount_Existing Approach",
        #             "amount_Default",
        #             *[f"amount_{e}" for e in _exp_columns],
        #         ],
        #         st.session_state['period_to_date'][st.session_state['period']],
        #         rename=False,
        #     )
        #
        #     res = pd.concat([list(d.values())[0].set_index(["record_id", "date"]) for d in forecast_results],
        #                     axis=1).reset_index()
        #     res.date = res.date.dt.date
        #     res = res.merge(st.session_state["sl_df"], left_on="record_id", right_on="ASSOCIATED_DEAL_IDS", how="left")
        #     amount_columns = [c for c in res.columns if c.startswith("amount")]
        #     for col in amount_columns:
        #         res[col] = res[col] * res["PROP"]
        #     if st.session_state["sl_selected"] != "Overall":
        #         res = res[res["NAME"] == st.session_state["sl_selected"]]
        #     res = res.groupby("date")[amount_columns].sum().reset_index()
        #
        #     fig = go.Figure()
        #
        #     for i, experiment_name in enumerate(["Current"] + st.session_state["reporting-experiments"]):
        #         if experiment_name == "Current":
        #             if len(st.session_state["forecast_results"].get("Current", {})) == 0:
        #                 continue
        #         fig.add_trace(
        #             go.Scatter(
        #                 x=res["date"],
        #                 y=res[f"amount_{experiment_name}"],
        #                 mode="lines",
        #                 name=experiment_name,
        #                 line={
        #                     "color": colors.DEFAULT_PLOTLY_COLORS[i + 1 % 10]
        #                 },
        #             )
        #         )
        #
        #     # Customize layout
        #     fig.update_layout(
        #         title="Future Prediction",
        #         xaxis_title="Date",
        #         yaxis_title="Amount (in $)",
        #         # legend_title_text='Features'
        #     )
        #
        #     # Show the plot
        #     st.plotly_chart(fig)
        #
        #     st.table(
        #         res.rename(
        #             columns={
        #                 "amount": "Existing",
        #                 "date": "Date",
        #                 **{
        #                     f"amount_{e}": e
        #                     for e in st.session_state["reporting-experiments"]
        #                 },
        #             }
        #         )
        #         .style.set_table_styles(
        #             [
        #                 {
        #                     "selector": "thead  th",
        #                     "props": "background-color: #2d7dce; text-align:center; color: white;font-size:0.9rem;border-bottom: 1px solid black !important;",
        #                 },
        #                 {
        #                     "selector": "tbody  th",
        #                     "props": "font-weight:bold;font-size:0.9rem;color:#000;",
        #                 },
        #                 {
        #                     "selector": "tbody  tr:nth-child(2n + 2)",
        #                     "props": "background-color: #aacbec;",
        #                 },
        #             ],
        #             overwrite=False,
        #         )
        #         .format(precision=0, thousands=",")
        #     )
        #
        # with st.expander("Future Report"):
        #     st.session_state["future_df_"] = st.session_state["future_df"].copy()
        #     experiment_name = st.session_state['selected_report_experiment']
        #     _exp_columns = []
        #
        #     if experiment_name in st.session_state["cohort_information"]:
        #         cohort_info = st.session_state["cohort_information"][experiment_name]
        #         cohort_df = cohort_info["cohort_df"]
        #         cohort_selected_features = cohort_info["cohort_selected_features"]
        #
        #         st.session_state["future_df_"][cohort_selected_features] = st.session_state["future_df_"][
        #             cohort_selected_features].fillna("None")
        #         st.session_state["future_df_"] = st.session_state["future_df_"].merge(
        #             cohort_df[cohort_selected_features + ["eff_probability"]],
        #             on=cohort_selected_features,
        #             how="left"
        #         ).rename(columns={"eff_probability": f"{experiment_name}_prob"})
        #
        #         if not cohort_df.iloc[0]["selected"]:
        #             ids = st.session_state["future_df_"][
        #                 st.session_state["future_df_"]['deal_probability'] != st.session_state["future_df_"][
        #                     'effective_probability']].index
        #             st.session_state["future_df_"].loc[ids, f"{experiment_name}_prob"] = \
        #                 st.session_state["future_df_"].loc[ids, 'effective_probability']
        #
        #         _exp_columns.append(experiment_name)
        #
        #     forecast_results = calculate_forecasts(
        #         st.session_state["future_df_"],
        #         ["final_probability", "deal_probability"] + [f"{e}_prob" for e in _exp_columns],
        #         ["amount_Existing Approach", "amount_Default"] + [f"amount_{e}" for e in _exp_columns],
        #         st.session_state['period_to_date'][st.session_state['period']],
        #         rename=False
        #     )
        #
        #     res = pd.concat([list(d.values())[0].set_index(["record_id", "date"]) for d in forecast_results],
        #                     axis=1).reset_index()
        #     res.date = res.date.dt.date
        #     res = res.merge(st.session_state["sl_df"], left_on="record_id", right_on="ASSOCIATED_DEAL_IDS", how="left")
        #     amount_columns = [c for c in res.columns if c.startswith("amount")]
        #     for col in amount_columns:
        #         res[col] = res[col] * res["PROP"]
        #
        #     res_ = res.merge(
        #         st.session_state['future_df_'][
        #             ['record_id', 'customer_segment', 'associated_company', 'deal_name', 'deal_stage', 'pipeline']],
        #         on='record_id',
        #         how='left'
        #     ).fillna('Unassigned')
        #
        #     map_deal_pipeline = {
        #         '0_new': 'Pipeline 0-2', '1_connected_to_meet': 'Pipeline 0-2', '2_needs_expressed': 'Pipeline 0-2',
        #         '3_qualified_oppurtunity': 'Pipeline 3-6', '4_proposal_presented': 'Pipeline 3-6',
        #         '5_verbal_agreement': 'Pipeline 3-6',
        #         '6_contracting': 'Pipeline 3-6', '7_closed_won': '7_closed_lost', '7_closed_lost': 'Backlog'
        #     }
        #     res_['Category'] = res_['deal_stage'].map(map_deal_pipeline)
        #     st.session_state['res_'] = res_.copy()
        #     if experiment_name in ['Existing Approach', 'Default'] or experiment_name in st.session_state["cohort_information"]:
        #         pivot_res = res_.pivot_table(
        #             index=['Category', 'pipeline', 'associated_company', 'record_id', 'deal_name', 'deal_stage',
        #                    'customer_segment',
        #                    'NAME'],
        #             columns='date',
        #             values=f'amount_{experiment_name}',
        #             aggfunc='sum'
        #         ).reset_index().fillna(0)
        #
        #         pivot_res.columns = [
        #             pd.to_datetime(col, errors='coerce') if col not in ['Category', 'pipeline', 'associated_company',
        #                                                                 'record_id',
        #                                                                 'deal_name',
        #                                                                 'deal_stage', 'customer_segment',
        #                                                                 'NAME', ]
        #             else col for col in pivot_res.columns]
        #         date_columns = [col for col in pivot_res.columns if isinstance(col, pd.Timestamp)]
        #         years = sorted(set(col.year for col in date_columns))
        #
        #         cumulative_sum_df = pivot_res.copy()
        #         for year in years:
        #             year_columns = [col for col in date_columns if col.year == year]
        #             cumulative_sum_df[f'Cumulative_{year}'] = pivot_res[year_columns].sum(axis=1)
        #         updated_date_columns = [col.strftime('%b-%y') for col in date_columns]
        #         cumulative_sum_df.rename(columns={old:new for old,new in zip(date_columns, updated_date_columns)}, inplace=True)
        #         st.session_state['final_all_sum_df'] = cumulative_sum_df.copy()
        #
        #         client_view = cumulative_sum_df.drop(columns=['pipeline'])
        #         client_view = client_view.rename(columns={
        #             'associated_company': 'COMPANY', 'record_id': 'HUBSPOT_ID', 'deal_name': "DEAL",
        #             'deal_stage': 'STAGE', 'customer_segment': "SEGMENT", 'NAME': 'SERVICE'
        #         })
        #
        #         st.markdown(f"<div class='selected-snapshot'>Selected Experiment for Report : {experiment_name}</div>",
        #                     unsafe_allow_html=True)
        #         filename = st.session_state['filename']
        #         if filename:
        #             if not filename.endswith(".xlsx"):
        #                 filename += ".xlsx"
        #             buffer = BytesIO()
        #             with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        #                 client_view.to_excel(writer, index=False, sheet_name='Report')
        #                 final_all_sum_df = st.session_state['final_all_sum_df'].copy()
        #                 depts = final_all_sum_df['NAME'].unique()
        #                 for dept in depts:
        #                     dept_df = final_all_sum_df[final_all_sum_df['NAME'] == dept]
        #                     dept_df.to_excel(writer, sheet_name=dept, index=False)
        #                 writer.close()
        #             st.download_button(label="Download Report", data=buffer.getvalue(), file_name=filename,
        #                                mime="application/vnd.ms-excel", key='download_report')
        #         st.table(
        #             client_view.style.set_table_styles
        #                 (
        #                 [
        #                     {
        #                         "selector": "thead  th",
        #                         "props": "background-color: #2d7dce; text-align:center; color: white;font-size:0.9rem;border-bottom: 1px solid black !important;",
        #                     },
        #                     {
        #                         "selector": "tbody  th",
        #                         "props": "font-weight:bold;font-size:0.9rem;color:#000;",
        #                     },
        #                 ],
        #                 overwrite=False,
        #             )
        #             .format(precision=0, thousands=",")
        #         )