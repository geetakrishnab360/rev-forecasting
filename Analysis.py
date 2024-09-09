import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from scipy.optimize import minimize
from snowflake_utils import (
    convert_period_to_dates,
    convert_dates_to_string,
    fetch_data_from_db,
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

# from db import fetch_experiment
# from data_utils import convert_to_cohort_df

st.set_page_config(layout="wide", page_title="Blend Forecasting")

load_dotenv()
prepare_actual_rev_data()
set_header("Forecasting Simulation - Pipeline Analysis")
create_database()


# data = fetch_experiment("geeta", "default")
# print(data)
# print(convert_to_cohort_df(data))


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


def change_experiment():
    st.session_state["selected_experiment_name"] = st.session_state[
        "selected-experiment"
    ]

    if st.session_state["selected_experiment_name"] != "Existing Approach":
        exp_df = convert_to_cohort_df(
            fetch_experiment(
                "karthik", st.session_state["selected_experiment_name"]
            )
        )
        st.session_state["cohort_selected_features"] = list(
            exp_df.drop(["cohort", "eff_probability"], axis=1).columns
        )
        st.session_state["cohort_df"] = exp_df.copy()
        # st.write(st.session_state["cohort_df"])
        st.session_state["active_df"] = st.session_state["data_df"].copy()
        st.session_state["active_df"][st.session_state.get("cohort_selected_features", [])] = \
            st.session_state["active_df"][st.session_state.get("cohort_selected_features", [])].fillna('None')

        st.session_state["active_df"] = (
            st.session_state["active_df"]
            .merge(
                st.session_state["cohort_df"],
                on=st.session_state.get("cohort_selected_features", []), how="left"
            )
            .copy()
        )

        # st.write(st.session_state.get("cohort_selected_features", []))
        # temp_cohort_df = st.session_state["cohort_df"].copy()
        # temp_cohort_df[st.session_state.get("cohort_selected_features", [])] = temp_cohort_df[st.session_state.get("cohort_selected_features", [])].fillna('Unknown')
        # st.write(temp_cohort_df[st.session_state.get("cohort_selected_features", [])].isna().sum())
        #
        # st.session_state["active_df"] = (
        #     st.session_state["active_df"]
        #     .merge(
        #         temp_cohort_df,
        #         on=st.session_state.get("cohort_selected_features", []), how="left"
        #     )
        #     .copy()
        # )
        #
        # st.session_state["active_df"][st.session_state.get("cohort_selected_features", [])] = st.session_state["active_df"][
        #     st.session_state.get("cohort_selected_features", [])].replace('Unknown', None)

        st.session_state["active_df"]["final_probability"] = st.session_state[
            "active_df"
        ]["eff_probability"].fillna(0)
        # st.write(st.session_state["active_df"])
        forecast_results = calculate_forecasts(
            st.session_state["active_df"], "2024-07-01"
        )

        st.session_state["forecast_results"][
            st.session_state["selected_experiment_name"]
        ] = forecast_results.copy()


def default_experiment():
    # if len(selected_features) == 0:
    #     st.session_state["default_cohort_df"] = pd.DataFrame()
    #     return
    # st.session_state["default_cohort_df"] = pd.DataFrame()
    default_cohort_df = pd.DataFrame()
    default_cohort_df['deal_stage'] = (
        st.session_state["data_df"]['deal_stage']
        .drop_duplicates()
        .reset_index(drop=True)
    )

    num_cohorts = default_cohort_df.shape[0]
    default_cohort_df['cohort'] = [f"cohort {i}" for i in range(1, num_cohorts + 1)]
    # st.write("Before adding prob", cohort_df)
    # if 'deal_stage' in selected_features:
    deal_stage_and_probability_dict = st.session_state['data_df'].set_index('deal_stage')[
        'deal_probability'].to_dict()
    # st.write(default_cohort_df)
    default_cohort_df["eff_probability"] = default_cohort_df['deal_stage'].map(
        deal_stage_and_probability_dict
    )
    st.session_state["default_df"] = st.session_state["data_df"].copy()
    st.session_state["default_df"] = (
        st.session_state["default_df"]
        .merge(
            default_cohort_df,
            on='deal_stage', how="left"
        )
        .copy()
    )
    # else:
    #     selected_features_and_probability = st.session_state['data_df'][selected_features + ['final_probability']]
    #     t = selected_features_and_probability.groupby(selected_features,
    #                                                   dropna=False).final_probability.mean().reset_index()
    #     default_cohort_df = default_cohort_df.merge(t, on=selected_features, how='left')
    #     default_cohort_df.rename(columns={'final_probability': 'eff_probability'}, inplace=True)
    # st.write(cohort_df)

    st.session_state["default_cohort_df"] = default_cohort_df[
        ["cohort", 'deal_stage', "eff_probability"]
    ]

    # st.session_state["experiment"] = "Default"
    st.session_state["default_df"] = st.session_state["data_df"].copy()
    st.session_state["default_df"] = (
        st.session_state["default_df"]
        .merge(
            st.session_state["default_cohort_df"],
            on='deal_stage', how="left"
        )
        .copy()
    )
    st.session_state["default_df"]["final_probability"] = st.session_state[
        "default_df"
    ]["eff_probability"].fillna(0)

    forecast_results = calculate_forecasts(
        st.session_state["default_df"], "2024-07-01"
    )

    st.session_state["forecast_results"]["Default"] = forecast_results.copy()
    # pass


def pull_data():
    if st.session_state.get("enter-custom-dates", False):
        pass
    else:
        start_date, end_date = convert_period_to_dates(
            st.session_state.get("data_period", 6)
        )

        # print(start_date, end_date)

    st.session_state["dates_string"] = convert_dates_to_string(
        start_date, end_date
    )
    # print(st.session_state["dates_string"])
    data = fetch_data_from_db(st.session_state["dates_string"])

    st.session_state["hubspot_id_dict"], st.session_state["data_df"] = (
        preprocess(data)
    )

    if "forecast_results" in st.session_state:
        del st.session_state["forecast_results"]


def cohort_generate_forecast(experiment_name="Current"):
    # st.write(st.session_state["cohort_df"])
    st.session_state['cohort_df'] = edited_data.copy()
    st.session_state["active_df"] = st.session_state["data_df"].copy()
    st.session_state["active_df"][st.session_state.get("cohort_selected_features", [])] = st.session_state["active_df"][
        st.session_state.get("cohort_selected_features", [])].fillna('None')
    st.session_state["active_df"] = (
        st.session_state["active_df"]
        .merge(
            edited_data,
            on=st.session_state.get("cohort_selected_features", []), how="left"
        )
        .copy()
    )
    # st.write(st.session_state["active_df"])
    st.session_state["active_df"]["final_probability"] = st.session_state[
        "active_df"
    ]["eff_probability"].fillna(0)

    forecast_results = calculate_forecasts(
        st.session_state["active_df"], "2024-07-01"
    )

    st.session_state["forecast_results"]["Current"] = forecast_results.copy()
    st.session_state["update_probabilities_clicked"] = True  # Set to True when button is clicked


# Initialize the session state variable if not already present
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


def cohorts_update(selected_features):
    if len(selected_features) == 0:
        st.session_state["cohort_df"] = pd.DataFrame()
        return
    cohort_df = (
        st.session_state["data_df"][selected_features]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    num_cohorts = cohort_df.shape[0]
    cohort_df["cohort"] = [f"cohort {i}" for i in range(1, num_cohorts + 1)]
    # st.write("Before adding prob", cohort_df)
    if 'deal_stage' in selected_features:
        deal_stage_and_probability_dict = st.session_state['data_df'].set_index('deal_stage')[
            'deal_probability'].to_dict()
        cohort_df["eff_probability"] = cohort_df["deal_stage"].map(
            deal_stage_and_probability_dict
        )
    else:
        selected_features_and_probability = st.session_state['data_df'][selected_features + ['final_probability']]
        t = selected_features_and_probability.groupby(selected_features,
                                                      dropna=False).final_probability.mean().reset_index()
        cohort_df = cohort_df.merge(t, on=selected_features, how='left')
        cohort_df.rename(columns={'final_probability': 'eff_probability'}, inplace=True)
        # st.write(cohort_df)

    st.session_state["cohort_df"] = cohort_df[
        ["cohort", *selected_features, "eff_probability"]
    ]
    st.session_state["cohort_df"] = st.session_state["cohort_df"].fillna("None")
    # st.write(st.session_state["cohort_df"])
    st.session_state["disable_calculations"] = False
    enable_save_experiment()


def allow_calculation():
    st.session_state["disable_calculations"] = False


def calculate_current_vs_act_error(eff_probabilities):
    st.session_state['cohort_df']['eff_probability'] = eff_probabilities
    edited_data = st.session_state['cohort_df'].copy()
    st.session_state["active_df"] = st.session_state["data_df"].copy()
    st.session_state["active_df"][st.session_state.get("cohort_selected_features", [])] = st.session_state["active_df"][
        st.session_state.get("cohort_selected_features", [])].fillna('None')
    st.session_state["active_df"] = (
        st.session_state["active_df"]
        .merge(
            edited_data,
            on=st.session_state.get("cohort_selected_features", []), how="left"
        )
        .copy()
    )
    # st.write(st.session_state["active_df"])
    st.session_state["active_df"]["final_probability"] = st.session_state[
        "active_df"
    ]["eff_probability"].fillna(0)

    forecast_results = calculate_forecasts(
        st.session_state["active_df"], "2024-07-01"
    )

    st.session_state["forecast_results"]["Current"] = forecast_results.copy()

    # cohort_generate_forecast()  # Update the forecast with new probabilities

    # Recalculate agg_df
    agg_df = pd.DataFrame()
    column_order = []
    for experiment_name in list(st.session_state["forecast_results"].keys()):
        snapshot_numbers = _calculate_snapshot_monthly_number(
            st.session_state["dates_string"], experiment_name
        )
        if len(agg_df) == 0:
            agg_df = (
                aggregate_snapshot_numbers(snapshot_numbers)
                .rename(
                    columns={
                        "Forecast": experiment_name,
                        "MAPE": f"{experiment_name} Vs Act (%Error)"
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
                        "Forecast": experiment_name,
                        "MAPE": f"{experiment_name} Vs Act (%Error)"
                    }
                )
                .copy(),
                left_index=True,
                right_index=True,
            )
        column_order.extend(
            [experiment_name, f"{experiment_name} Vs Act (%Error)"]
        )
    agg_df = agg_df[["Actual"] + column_order]
    # print(agg_df["Current Vs Act (%Error)"].mean())
    # Return the mean of the Current Vs Act (%Error) column
    return agg_df["Current Vs Act (%Error)"].mean()


def optimize_eff_probability(max_iter, progress_bar, progress_text):
    initial_guess = st.session_state['cohort_df']['eff_probability'].values
    bounds = [(0, 1) for _ in range(len(initial_guess))]
    print("Before optimization: ", initial_guess)
    print(calculate_current_vs_act_error(initial_guess))

    def progress_callback(xk):
        progress = min(1.0, progress_callback.iteration / max_iter)
        progress_bar.progress(progress)
        progress_text.text(f"Optimization progress: {int(progress * 100)}%")
        progress_callback.iteration += 1

    progress_callback.iteration = 0

    result = minimize(
        calculate_current_vs_act_error,
        initial_guess,
        bounds=bounds,
        method='L-BFGS-B',
        options={
            'maxiter': max_iter,
            'ftol': 1e-4,
            'disp': True
        },
        callback=progress_callback
    )

    if result.success:
        optimized_eff_probabilities = result.x
        print("After optimization: ", optimized_eff_probabilities)
        st.session_state['cohort_df']['eff_probability'] = optimized_eff_probabilities
        st.success("Optimization successful!")
    else:
        st.error("Optimization failed")

    progress_bar.progress(1.0)
    progress_text.text("Optimization complete")


with open("./styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )

if "cohort_df" not in st.session_state:
    st.session_state["cohort_df"] = pd.DataFrame()

if "data_df" not in st.session_state:
    pull_data()

if "all_experiments" not in st.session_state:
    st.session_state["all_experiments"] = fetch_all_experiments("karthik")

if "forecast_results" not in st.session_state:
    st.session_state["forecast_results"] = {}

    forecast_results = calculate_forecasts(
        st.session_state["data_df"], "2024-07-01"
    )

    st.session_state["forecast_results"][
        "Existing Approach"
    ] = forecast_results.copy()
    default_experiment()

    # print(st.session_state["forecast_results"])

if "selected_experiment_name" not in st.session_state:
    st.session_state["selected_experiment_name"] = "Existing Approach"

# left_pane, main_pane, right_pane = st.columns((0.2, 0.6, 0.2))
left_pane, main_pane = st.columns((0.25, 0.75))

with left_pane:
    # Dropdown for period selection
    if not st.session_state.get("enter-custom-dates", False):
        st.selectbox(
            "Data Period",
            options=[1, 3, 6],
            key="data_period",
            format_func=format_period_text,
            index=2,
        )
    else:
        st.text_input(
            "Data Period",
            placeholder="type in simple english",
            key="data_period_text",
        )

    st.checkbox(
        "Enter Custom Dates",
        key="enter-custom-dates",
    )

    st.markdown(
        "<div class='date-period-text'>Selected Period : Jun 2024</div>",
        unsafe_allow_html=True,
    )

    st.button("Fetch Data", on_click=pull_data)

    st.divider()

    st.selectbox(
        "Load Experiment",
        options=["Existing Approach"] + st.session_state["all_experiments"],
        key="selected-experiment",
        on_change=change_experiment,
    )

    st.multiselect(
        "Select Experiments for Reporting",
        options=st.session_state["all_experiments"],
        key="reporting-experiments",
    )

    st.divider()

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
    with _columns[1]:
        st.button("Copy Experiment")

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

        max_iter = st.number_input("Max Iterations", min_value=1, value=10, step=1, key="max_iter",
                                   label_visibility="collapsed")
        if st.button("Optimize Probabilities", disabled=not st.session_state.get("update_probabilities_clicked", False),
                     type="secondary"):
            progress_bar = st.progress(0)
            progress_text = st.empty()
            optimize_eff_probability(max_iter, progress_bar, progress_text)

        progress_bar = st.progress(0)
        progress_text = st.empty()

    st.divider()
    with st.container():

        st.subheader("Results")
        agg_df = pd.DataFrame()
        column_order = []
        for expriment_name in list(
                st.session_state["forecast_results"].keys()
        ):
            # print("###, ", expriment_name)
            # st.session_state[
            #     "reporting-experiments"
            # ]:
            snapshot_numbers = _calculate_snapshot_monthly_number(
                st.session_state["dates_string"], expriment_name
            )
            # print(snapshot_numbers)
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
            # print(agg_df)
        agg_df = agg_df[["Actual"] + column_order]
        print(agg_df)
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
        for experiment in list(st.session_state["forecast_results"].keys()):
            # st.session_state[
            #     "reporting-experiments"
            # ]:
            fig.add_trace(
                go.Scatter(
                    x=agg_df.index,
                    y=agg_df[experiment],
                    mode="lines+markers",
                    name=experiment,
                ),
                row=1,
                col=1,
            )

            # Add bar plot for MAPE (%Error)
            fig.add_trace(
                go.Bar(
                    x=agg_df.index,
                    y=agg_df[f"{experiment} Vs Act (%Error)"],
                    name=f"{experiment} MAPE",
                ),
                row=1,
                col=2,
            )

        # If there are more experiments, add stacked bars here:
        # fig.add_trace(
        #     go.Bar(x=df.columns, y=df.loc['Experiment MAPE'], name='Experiment MAPE'),
        #     row=2, col=1
        # )

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
            st.selectbox(
                "Experiment",
                options=set(
                    list(st.session_state["forecast_results"].keys())
                    + st.session_state["all_experiments"]
                ),
                key="experiment-snapshot-results",
            )
            # st.write(st.session_state["experiment-snapshot-results"])
            snapshot_numbers = _calculate_snapshot_monthly_number(
                st.session_state["dates_string"],
                st.session_state["experiment-snapshot-results"],
            )

            for date, res in snapshot_numbers.items():
                res = res.T
                ren_cols = []
                for i, cols in enumerate(res.columns):
                    ren_cols.append((date + pd.DateOffset(months=i + 1)).strftime("%b"))
                res.columns = ren_cols
                # print("start")
                # print(ren_cols)
                # print("end")
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

        with st.expander("Cohort Results"):
            st.selectbox(
                "Experiment",
                options=set(
                    list(st.session_state["forecast_results"].keys())
                    + st.session_state["all_experiments"]
                ),
                key="experiment-cohort-results",
            )
            # st.write(st.session_state["dates_string"])
            coh_res = calculate_cohort_error(st.session_state["dates_string"],
                                             st.session_state["experiment-cohort-results"])
            if coh_res is not None:
                st.table(
                    coh_res.set_index('cohort').style.set_table_styles(
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
                                "selector": "tbody  tr:nth-child(2n + 2)",
                                "props": "background-color: #aacbec;",
                            },
                            {
                                "selector": "tbody  tr:nth-child(2n + 2) > td,tbody  tr:nth-child(2n + 3) > th",
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
                # st.write(st.session_state['cohort_df'])
            else:
                st.write("No Cohort Data available")
