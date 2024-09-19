import streamlit as st
import plotly.graph_objs as go
from io import BytesIO
from plotly.subplots import make_subplots
from plotly import colors
from dotenv import load_dotenv
from scipy.optimize import minimize
from datetime import datetime
from Analysis import main_pane, left_pane
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
    aggregate_snapshot_numbers, create_exo_df,
)

from db import (
    create_database,
    insert_experiment_into_db,
    fetch_all_experiments,
    fetch_experiment, create_bu_database, insert_bu_into_db, delete_bu_from_db, fetch_all_bu_data, update_bu_in_db,
)

from components import set_header
import numpy as np
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Top Down", initial_sidebar_state="collapsed")
load_dotenv()
set_header("Top Down Forecasting")
create_bu_database()

with open("./styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )

if 'is_fetched' not in st.session_state and 'is_bu_inserted' in st.session_state and st.session_state['is_bu_inserted']:
    st.session_state['all_bu_dfs'] = fetch_all_bu_data('karthik')
    st.session_state['is_fetched'] = True
# st.write(st.session_state['all_bu_dfs'].dtypes)

left_pane, main_pane = st.columns([5, 1])
with left_pane:
    # col1, col2, col3 = st.columns(3)
    #
    # with col1:
    #     selected_bu = st.selectbox('Select Business Unit', st.session_state['all_bu_dfs']['bu'].unique())
    #
    # with col2:
    #     selected_date = st.date_input('Select Date')
    #     selected_date = pd.to_datetime(selected_date)
    #
    # with col3:
    #     selected_y = st.number_input('Enter Value for y', min_value=0.0)
    #
    # # st.write('Selected Business Unit:', selected_bu)
    # # st.write('Selected Date:', selected_date)
    # # st.write('Selected Value for y:', selected_y)
    # # st.write(st.session_state['all_bu_dfs'].dtypes)
    # # st.write(st.session_state['all_bu_dfs'].tail())
    # # st.write(st.session_state['all_bu_dfs'].loc[(st.session_state['all_bu_dfs']['bu'] == selected_bu)])
    # # st.write(st.session_state['all_bu_dfs'].loc[st.session_state['all_bu_dfs']['ds'] == str(selected_date)])
    #
    # cols = st.columns((0.5, 1))
    #
    # with cols[0]:
    #     if st.button('Insert Data'):
    #         if not st.session_state['all_bu_dfs'].loc[(st.session_state['all_bu_dfs']['bu'] == selected_bu) & (
    #                 st.session_state['all_bu_dfs']['ds'] == str(selected_date))].empty:
    #             st.write('Data already exists. Please update the data')
    #             # st.stop()
    #         else:
    #             new_data = pd.DataFrame({'bu': [selected_bu], 'ds': [str(selected_date)], 'y': [selected_y]})
    #             insert_bu_into_db(new_data, 'karthik')
    #             new_data['y'] = new_data['y'].astype(float)
    #             st.session_state['all_bu_dfs'] = pd.concat([st.session_state['all_bu_dfs'], new_data])
    #             st.write('Data Inserted Successfully')
    #
    # with cols[1]:
    #     if st.button('Update Data'):
    #         if st.session_state['all_bu_dfs'].loc[(st.session_state['all_bu_dfs']['bu'] == selected_bu) & (
    #                 st.session_state['all_bu_dfs']['ds'] == str(selected_date))].empty:
    #             st.write('Data does not exist. Please insert the data first')
    #             # st.stop
    #         else:
    #             new_data = pd.DataFrame({'bu': [selected_bu], 'ds': [str(selected_date)], 'y': [selected_y]})
    #             update_bu_in_db(new_data, 'karthik')
    #             st.session_state['all_bu_dfs'].loc[(st.session_state['all_bu_dfs']['bu'] == selected_bu) & (
    #                     st.session_state['all_bu_dfs']['ds'] == str(selected_date)), 'y'] = selected_y
    #             st.write('Data Updated Successfully')

    if 'edit_bu' not in st.session_state:
        st.session_state['edit_bu'] = pd.DataFrame(columns=['BU','Date','Revenue'])
        st.session_state['edit_bu']['BU'] = st.session_state['all_bu_dfs']['bu'].unique()
        today = datetime(2024,7,1)
        st.session_state['edit_bu']['Date'] =  today
        st.session_state['edit_bu']['Revenue'] = 0.0
        st.session_state['edit_bu']['selected'] = True

    column_config = {
        "BU": st.column_config.TextColumn(
            "BU",
            disabled=True,
        ),
        "Date": st.column_config.DateColumn(
            "Date",
            disabled=False,
        ),
        "Revenue": st.column_config.NumberColumn(
            "Revenue",
            disabled=False,
            required=True,
            min_value=0,
            step=0.01,
        ),
    }

    st.text("Edit BU data:")
    edited_data = st.data_editor(
        st.session_state["edit_bu"],
        key="edited_bu_data",
        hide_index=True,
        column_config=column_config,
    )
    st.session_state['edit_bu'] = edited_data

    if st.button('Save Edited Data'):
            selected_data = st.session_state['edit_bu'][st.session_state['edit_bu']['selected']]
            selected_data = selected_data.rename(columns={'BU': 'bu', 'Date': 'ds', 'Revenue': 'y'})
            selected_data['y'] = selected_data['y'].astype(float)
            selected_data['ds'] = selected_data['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
            if len(selected_data) != 0:
                selected_data = selected_data[['bu','ds','y']]
                for i, row in selected_data.iterrows():
                    # row = pd.DataFrame(row).T
                    # row = row.reset_index(drop=True)
                    # st.write(row)
                    # st.write(row.dtypes)
                    # st.write(st.session_state['all_bu_dfs'].loc[(st.session_state['all_bu_dfs']['bu'] == row['bu'])])
                    # st.write(st.session_state['all_bu_dfs'].loc[(st.session_state['all_bu_dfs']['bu'] == row['bu']) & (
                    #         st.session_state['all_bu_dfs']['ds'] == row['ds'])])
                    if len(st.session_state['all_bu_dfs'].loc[(st.session_state['all_bu_dfs']['bu'] == row['bu']) & (
                            st.session_state['all_bu_dfs']['ds'] == row['ds'])]) != 0:
                        row_c = pd.DataFrame(row).T
                        row_c = row_c.reset_index(drop=True)
                        update_bu_in_db(row_c, 'karthik')
                        st.session_state['all_bu_dfs'].loc[(st.session_state['all_bu_dfs']['bu'] == row['bu']) & (
                            st.session_state['all_bu_dfs']['ds'] == row['ds']), 'y'] = row['y']
                    else:
                        row_c = pd.DataFrame(row).T
                        row_c = row_c.reset_index(drop=True)
                        insert_bu_into_db(row_c, 'karthik')
                        st.session_state['all_bu_dfs'] = pd.concat([st.session_state['all_bu_dfs'], row_c])
                        st.session_state['all_bu_dfs'] = st.session_state['all_bu_dfs'].reset_index(drop=True)
                        # st.write('Data Inserted Successfully')
    # st.write(st.session_state['all_bu_dfs'].tail())
    with st.expander('Business Units Data', expanded=True):
        st.selectbox('Select Business Unit',
                     options=['Overall'] + st.session_state['all_bu_dfs']['bu'].unique().tolist(),
                     key='bu_graph_selected', index=0)

        bu_data = st.session_state['all_bu_dfs'].copy()
        bu_data['ds'] = pd.to_datetime(bu_data['ds'])
        bu_data['ds'] = bu_data['ds'].dt.date
        bu_data['y'] = bu_data['y'].astype(float)
        bu_filename = 'BU_data.xlsx'
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            bu_data.to_excel(writer, index=False, sheet_name='Overall')
            bus = bu_data['bu'].unique()
            for bu in bus:
                dept_df = bu_data[bu_data['bu'] == bu]
                dept_df.to_excel(writer, sheet_name=bu, index=False)
            writer.close()
        st.download_button(label="Download Data", data=buffer.getvalue(), file_name=bu_filename,
                           mime="application/vnd.ms-excel", key='download_bu_data')

        fig = go.Figure()
        if st.session_state['bu_graph_selected'] == "Overall":
            all_bu_dfs = st.session_state['all_bu_dfs'].copy()
            for bu in all_bu_dfs['bu'].unique():
                bu_data = all_bu_dfs[all_bu_dfs['bu'] == bu]
                fig.add_trace(
                    go.Scatter(
                        x=bu_data['ds'],
                        y=bu_data['y'],
                        mode='lines',
                        name=bu
                    )
                )
        else:
            bu_data = st.session_state['all_bu_dfs'].copy()
            bu_data = bu_data[bu_data['bu'] == st.session_state['bu_graph_selected']]
            fig.add_trace(
                go.Scatter(
                    x=bu_data['ds'],
                    y=bu_data['y'],
                    mode='lines',
                    name='Overall'
                )
            )
        fig.update_layout(
            title=f'{st.session_state['bu_graph_selected']} Business Units Graph',
            xaxis_title='Date',
            yaxis_title='Value',
            legend_title='Business Units'
        )

        st.plotly_chart(fig)
        # st.table(
        #     bu_data.rename(
        #         columns={
        #             'bu': 'Business Unit',
        #             'ds': 'Date',
        #             'y': 'Value'
        #         }
        #     )
        #     .style.set_table_styles(
        #         [
        #             {
        #                 "selector": "thead  th",
        #                 "props": "background-color: #2d7dce; text-align:center; color: white;font-size:0.9rem;border-bottom: 1px solid black !important;",
        #             },
        #             {
        #                 "selector": "tbody  th",
        #                 "props": "font-weight:bold;font-size:0.9rem;color:#000;",
        #             },
        #         ],
        #         overwrite=False,
        #     )
        #     .format(precision=0, thousands=",")
        # )

        exo_df = create_exo_df()
    with st.expander('Exogenous Data', expanded=True):
        plot_show_df = exo_df.copy()
        plot_show_df = plot_show_df[['ds', 'S&P500', 'GOLD_PRICE', 'CRUDE_PRICE', 'CPI']]
        plot_show_df = plot_show_df[plot_show_df['ds'] >= pd.to_datetime('2022-01-01')]
        # i need editable table here
        if 'edit_exo_df' not in st.session_state:
            st.session_state['edit_exo_df'] = plot_show_df.copy()

        column_config = {
            "ds": st.column_config.DateColumn(
                "Date",
                disabled=True,
            ),
            "S&P500": st.column_config.NumberColumn(
                "S&P500",
                disabled=False,
                required=True,
                min_value=0,
                step=0.01,
            ),
            "GOLD_PRICE": st.column_config.NumberColumn(
                "GOLD_PRICE",
                disabled=False,
                required=True,
                min_value=0,
                step=0.01,
            ),
            "CRUDE_PRICE": st.column_config.NumberColumn(
                "CRUDE_PRICE",
                disabled=False,
                required=True,
                min_value=0,
                step=0.01,
            ),
            "CPI": st.column_config.NumberColumn(
                "CPI",
                disabled=False,
                required=True,
                min_value=0,
                step=0.01,
            ),
        }
        cols = st.columns((0.5, 1))
        with cols[0]:
            st.text("Edit Exogenous data:")
            edited_data = st.data_editor(
                st.session_state["edit_exo_df"],
                key="edited_exo_data",
                hide_index=True,
                column_config=column_config,
            )
        st.session_state['edit_exo_df'] = edited_data.copy()
        with cols[1]:
            fig = go.Figure()
            cols = ['S&P500', 'GOLD_PRICE', 'CRUDE_PRICE', 'CPI']
            for col in cols:
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state['edit_exo_df'].ds,
                        y=st.session_state['edit_exo_df'][col],
                        mode='lines',
                        name=col
                    )
                )
            fig.update_layout(
                title='Exogenous Data',
                xaxis_title='Date',
                yaxis_title='Value',
                legend_title='Exogenous Data'
            )

            st.plotly_chart(fig)



