import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly import colors
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
import numpy as np
import warnings

st.set_page_config(layout="wide", page_title="Forecast Results", initial_sidebar_state="collapsed")
set_header("Forecasting Simulation - Pipeline Analysis")
with open("./styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )
left_pane, main_pane = st.columns((0.25, 0.75))

with left_pane:
    st.multiselect(
        "Select Experiments for Comparison",
        options=["Existing Approach", "Default"]
                + st.session_state["all_experiments"],
        key="reporting-experiments",
    )
    st.selectbox(
        "Select Period",
        options=["Quarter", "6 Months", "1 Year"],
        key="period",
    )
    st.session_state['period_to_date'] = {"Quarter": "2024-10-01", "6 Months": "2025-01-01", "1 Year": "2025-07-01"}
with main_pane:
    if "sl_df" not in st.session_state:
        st.session_state["sl_df"] = pd.read_csv("sl_proportions.csv")
        st.session_state["sl_df"] = st.session_state["sl_df"][
            ~st.session_state["sl_df"].ASSOCIATED_DEAL_IDS.isna()
        ].copy()
        st.session_state["sl_df"][
            "ASSOCIATED_DEAL_IDS"
        ] = st.session_state["sl_df"]["ASSOCIATED_DEAL_IDS"].apply(
            lambda x: str(x).replace(".0", "")
        )
        st.session_state["sl_list"] = sorted(
            list(st.session_state["sl_df"].NAME.unique())
        )

    st.selectbox(
        "Service line",
        options=["Overall"] + st.session_state["sl_list"],
        key="sl_selected",
        index=0,
    )

    st.session_state["future_df_"] = st.session_state["future_df"]
    # st.write(st.session_state["future_df_"])
    _exp_columns = []
    for experiment_name in ["Current"] + st.session_state[
        "reporting-experiments"
    ]:
        if (
                len(
                    st.session_state["cohort_information"].get(
                        experiment_name, {}
                    )
                )
                > 0
        ):
            cohort_df = st.session_state["cohort_information"][
                experiment_name
            ]["cohort_df"]
            cohort_selected_features = st.session_state[
                "cohort_information"
            ][experiment_name]["cohort_selected_features"]
            st.session_state["future_df_"] = (
                st.session_state["future_df_"]
                .merge(
                    cohort_df[
                        [*cohort_selected_features, "eff_probability"]
                    ],
                    on=cohort_selected_features,
                    how="left",
                )
                .rename(
                    columns={
                        "eff_probability": f"{experiment_name}_prob"
                    }
                )
                .copy()
            )
            _exp_columns.append(experiment_name)

        if experiment_name == "Current":
            if (
                    len(
                        st.session_state["forecast_results"].get(
                            "Current", {}
                        )
                    )
                    == 0
            ):
                if experiment_name in _exp_columns:
                    _exp_columns.remove(experiment_name)
                continue

    forecast_results = calculate_forecasts(
        st.session_state["future_df_"],
        [
            "final_probability",
            "deal_probability",
            *[f"{e}_prob" for e in _exp_columns],
        ],
        [
            "amount_Existing Approach",
            "amount_Default",
            *[f"amount_{e}" for e in _exp_columns],
        ],
        st.session_state["period_to_date"][st.session_state["period"]],
        rename=False,
    )

    res = pd.concat(
        [
            list(d.values())[0].set_index(["record_id", "date"])
            for d in forecast_results
        ],
        axis=1,
    ).reset_index()
    res.date = res.date.dt.date
    res = res.merge(
        st.session_state["sl_df"],
        left_on="record_id",
        right_on="ASSOCIATED_DEAL_IDS",
        how="left",
    )
    amount_columns = [c for c in res.columns if c.startswith("amount")]
    for col in amount_columns:
        res[col] = res[col] * res["PROP"]
    if st.session_state["sl_selected"] != "Overall":
        res = res[res["NAME"] == st.session_state["sl_selected"]]
    res = res.groupby("date")[amount_columns].sum().reset_index()

    fig = go.Figure()

    for i, experiment_name in enumerate(
            ["Current"] + st.session_state["reporting-experiments"]
    ):
        if experiment_name == "Current":
            if (
                    len(
                        st.session_state["forecast_results"].get(
                            "Current", {}
                        )
                    )
                    == 0
            ):
                continue
        fig.add_trace(
            go.Scatter(
                x=res["date"],
                y=res[f"amount_{experiment_name}"],
                mode="lines",
                name=experiment_name,
                line={
                    "color": colors.DEFAULT_PLOTLY_COLORS[i + 1 % 10]
                },
            )
        )

    # Customize layout
    fig.update_layout(
        title="Future Prediction",
        xaxis_title="Date",
        yaxis_title="Amount (in $)",
        # legend_title_text='Features'
    )

    # Show the plot
    st.plotly_chart(fig)
    with st.expander("Future Results"):
        st.table(
            res.rename(
                columns={
                    "amount": "Existing",
                    "date": "Date",
                    **{
                        f"amount_{e}": e
                        for e in st.session_state["reporting-experiments"]
                    },
                }
            )
            .style.set_table_styles(
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
                ],
                overwrite=False,
            )
            .format(precision=0, thousands=",")
        )
