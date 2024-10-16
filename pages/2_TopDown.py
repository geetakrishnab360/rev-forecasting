import datetime as datet
import warnings
from io import BytesIO
import pandas as pd
import plotly.graph_objs as go
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
import pytz
from model_utilities import *
from new_db import *
from new_data_utils import *
import logging
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from prophet import Prophet
import boto3
import simplejson as json
import time
import uuid
from datetime import datetime
from components import (
    set_header,
    hide_sidebar,
)
from data_utils import (
    create_exo_df,
    update_df,
    record_changes,
    update_exo_df,
)
from new_db import (
    fetch_revenue_data_from_db,
    insert_new_rev_data,
    update_rev_data,
    fetch_forecast_data,
)
import streamlit as st

np.float_ = np.float64
warnings.filterwarnings("ignore")

aws_access_key_id = xxxxx
aws_secret_access_key = xxxxx
aws_session_token = xxxxx
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name='us-east-1'
)
lambda_client = session.client('lambda')
LAMBDA_FUNCTION_NAME = 'test-revforecasting1'


def invoke_lambda(user,
                  model_name,
                  bu,
                  request_id):
    payload = {'user': user,
               'model_name': model_name,
               'bu': bu,
               'request_id': request_id}
    response = lambda_client.invoke(
        FunctionName=LAMBDA_FUNCTION_NAME,
        InvocationType='Event',
        Payload=json.dumps(payload, ignore_nan=True)
    )
    return request_id


def train_models(user):
    bus = ['dsx', 'bts', 'emea', 'mlabs', 'fpai']
    uniq_id = str(uuid.uuid4())
    for bu in bus:
        req_id = invoke_lambda(user=user,
                               model_name=f'Prophet_{bu}',
                               bu=bu,
                               request_id=uniq_id)
        print(f"Lambda invocation triggered successfully for {bu} with request ID: {req_id}")
    return uniq_id


st.set_page_config(layout="wide", page_title="Top Down", initial_sidebar_state="collapsed")
hide_sidebar()
page = option_menu(
    menu_title=None,
    options=["Pipeline Analysis", "Pipeline Forecast", "TopDown Forecast"],
    default_index=2,
    orientation="horizontal",
    icons=["bar-chart-line-fill", "graph-up-arrow", "graph-up-arrow"],
    styles={
        "container": {
            # "background-color": "black",
            # "padding": "10px",
            # "margin": "10px 0px",
            # "font": "sans-serif",
            # "position": "relative",
            "border": "1px solid #d3d3d3",
            "border-radius": "5px",
            "margin": "0px 0px 0px 0px",
            "padding": "0px",
        },
        "nav-link": {
            "font-family": "Verdana, sans-serif",
            "font-size": "0.85rem",
            # "text-align": "left",
            "--hover-color": "grey",
            "--hover-background-color": "white",
            "margin": "0px 0px",
            "border-radius": "0px",
        },
        "nav-link-selected": {"background-color": "red", "color": "white"},
    },
)
if page == 'Pipeline Analysis':
    st.switch_page("Analysis.py")
if page == 'Pipeline Forecast':
    st.switch_page("pages/1_Forecast_Results.py")
set_header("Top Down Forecasting")

load_dotenv()

with open("./styles1.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )

if 'is_fetched' not in st.session_state:
    data = fetch_revenue_data_from_db()
    data = data.rename({"BUSINESS_UNIT": "bu", "DATE": "ds", "REVENUE": "y"}, axis=1)
    data = data[data['bu'] != 'test']
    data['ds'] = pd.to_datetime(data['ds'])
    data['ds'] = data['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
    data['y'] = data['y'].astype(float)
    st.session_state['all_bu_dfs'] = data.copy()
    st.session_state['is_fetched'] = True

latest_actual_date_time = st.session_state['all_bu_dfs']['ds'].max()
st.session_state['latest_actual_date_time'] = latest_actual_date_time
latest_actual_date = datet.datetime.strptime(latest_actual_date_time,
                                             "%Y-%m-%d %H:%M:%S").date() + relativedelta(months=+1)
st.session_state['latest_actual_date'] = latest_actual_date

left_pane, _ = st.columns([8, 1])
with left_pane:
    with st.expander('Data Inputs', expanded=False):
        if 'edit_bu' not in st.session_state:
            st.session_state['edit_bu'] = pd.DataFrame(columns=['BU', 'Date', 'Revenue'])
            st.session_state['edit_bu']['BU'] = st.session_state['all_bu_dfs']['bu'].unique()
            # today = datetime(2024, 7, 1)
            st.session_state['edit_bu']['Date'] = pd.to_datetime(latest_actual_date)
            st.session_state['edit_bu']['Revenue'] = 0.0
            st.session_state['edit_bu']['selected'] = False

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
            on_change=update_df,
        )
        st.markdown(
            "<h4 style='font-size: 0.8rem;'>Note: Date is always set to 1st of the input month by default</h4>",
            unsafe_allow_html=True,
        )

        selected_data = st.session_state['edit_bu'][st.session_state['edit_bu']['selected']]
        selected_data = selected_data.rename(columns={'BU': 'bu', 'Date': 'ds', 'Revenue': 'y'})
        selected_data['y'] = selected_data['y'].astype(float)
        selected_data['ds'] = selected_data['ds'].apply(lambda x: x.replace(day=1)).dt.strftime('%Y-%m-%d %H:%M:%S')
        st.session_state['is_update'] = False

        if not selected_data.empty:
            selected_data = selected_data[['bu', 'ds', 'y']]

            for i, row in selected_data.iterrows():
                existing_entry = st.session_state['all_bu_dfs'][
                    (st.session_state['all_bu_dfs']['bu'] == row['bu']) &
                    (st.session_state['all_bu_dfs']['ds'] == row['ds'])
                    ]

                if not existing_entry.empty:
                    cols = st.columns([0.4, 1])
                    cols[0].info("Are you sure you want to update the data?")
                    st.session_state['is_update'] = not st.button('Yes')
                    break

            if st.button('Save Edited Data', disabled=st.session_state['is_update']):
                selected_data = st.session_state['edit_bu'][st.session_state['edit_bu']['selected']]
                selected_data = selected_data.rename(columns={'BU': 'bu', 'Date': 'ds', 'Revenue': 'y'})
                selected_data['y'] = selected_data['y'].astype(float)
                selected_data['ds'] = selected_data['ds'].apply(lambda x: x.replace(day=1)).dt.strftime(
                    '%Y-%m-%d %H:%M:%S')

                if not selected_data.empty:
                    selected_data = selected_data[['bu', 'ds', 'y']]
                    for i, row in selected_data.iterrows():
                        existing_entry = st.session_state['all_bu_dfs'][
                            (st.session_state['all_bu_dfs']['bu'] == row['bu']) &
                            (st.session_state['all_bu_dfs']['ds'] == row['ds'])
                            ]
                        row_c = pd.DataFrame(row).T.reset_index(drop=True)

                        if not existing_entry.empty:
                            # update_bu_in_db(row_c, 'karthik')
                            update_rev_data(row_c, 'karthik')
                            st.session_state['all_bu_dfs'].loc[
                                (st.session_state['all_bu_dfs']['bu'] == row['bu']) &
                                (st.session_state['all_bu_dfs']['ds'] == row['ds']), 'y'] = row['y']
                        else:
                            # insert_bu_into_db(row_c, 'karthik')
                            insert_new_rev_data(row_c, 'karthik')
                            st.session_state['all_bu_dfs'] = pd.concat(
                                [st.session_state['all_bu_dfs'], row_c]).reset_index(
                                drop=True)
                    cols = st.columns([0.4, 1])
                    cols[0].success("Data saved successfully")

        latest_actual_date_time = st.session_state['all_bu_dfs']['ds'].max()
        st.session_state['latest_actual_date_time'] = latest_actual_date_time
        latest_actual_date = datet.datetime.strptime(latest_actual_date_time,
                                                     "%Y-%m-%d %H:%M:%S").date() + relativedelta(months=+1)
        st.session_state['latest_actual_date'] = latest_actual_date

        st.divider()

        st.header('Business Units Data')

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
        cols = st.columns((1, 1, 1))
        if 'bu_graph_selected' not in st.session_state:
            st.session_state['bu_graph_selected'] = "Overall"
        cols[0].selectbox('Select Business Unit',
                          options=['Overall'] + st.session_state['all_bu_dfs']['bu'].unique().tolist(),
                          key='bu_graph_selectedd',
                          index=(['Overall'] + st.session_state['all_bu_dfs']['bu'].unique().tolist()).index(
                              st.session_state['bu_graph_selected']),
                          on_change=record_changes, args=('bu_graph_selected', 'bu_graph_selectedd'))
        no_of_past_years = latest_actual_date + relativedelta(months=-1) - datet.datetime.strptime(
            st.session_state['all_bu_dfs']['ds'].min(), "%Y-%m-%d %H:%M:%S").date()
        # print(no_of_past_years)
        no_of_past_years = no_of_past_years.days // 365
        # print(datet.datetime.strptime(st.session_state['all_bu_dfs']['ds'].min(), "%Y-%m-%d %H:%M:%S").date())
        # print(latest_actual_date)
        if 'past_years' not in st.session_state:
            st.session_state['past_years'] = 2
        cols[1].selectbox('Select past years of actuals', options=range(1, no_of_past_years + 1), key='past_yearss',
                          index=(range(1, no_of_past_years + 1)).index(int(st.session_state['past_years'])),
                          on_change=record_changes, args=('past_years', 'past_yearss'))
        st.download_button(label="Download Data", data=buffer.getvalue(), file_name=bu_filename,
                           mime="application/vnd.ms-excel", key='download_bu_data')

        fig = go.Figure()
        if st.session_state['bu_graph_selected'] == "Overall":
            all_bu_dfs = st.session_state['all_bu_dfs'].copy()
            all_bu_dfs = all_bu_dfs[all_bu_dfs['ds'] >= str(pd.to_datetime(latest_actual_date_time) + relativedelta(
                years=-int(st.session_state['past_years'])))]
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
            bu_data = bu_data[bu_data['ds'] >= str(pd.to_datetime(latest_actual_date_time) + relativedelta(
                years=-int(st.session_state['past_years'])))]
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
            title=f"{st.session_state['bu_graph_selected']} Business Units Graph",
            xaxis_title='Date',
            yaxis_title='Value',
            legend_title='Business Units'
        )

        st.plotly_chart(fig)

        st.divider()

        st.header('Exogenous Data')

        exo_df = create_exo_df()
        plot_show_df = exo_df.copy()
        # plot_show_df = plot_show_df[['ds', 'S&P500', 'GOLD_PRICE', 'CRUDE_PRICE', 'CPI']]
        plot_show_df = plot_show_df[plot_show_df['ds'] >= pd.to_datetime('2022-01-01')]
        plot_show_df = plot_show_df.reset_index(drop=True)
        if 'edit_exo_df' not in st.session_state:
            st.session_state['edit_exo_df'] = plot_show_df.copy()

        column_config = {
            "ds": st.column_config.DateColumn(
                "Date",
                disabled=True,
            ),
            "FED_GRANTS": st.column_config.NumberColumn(
                "FED_GRANTS",
                disabled=False,
                required=True,
                min_value=0,
                step=0.01,
            ),
            "S&P500": st.column_config.NumberColumn(
                "S&P500",
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
            "GOLD_PRICE": st.column_config.NumberColumn(
                "GOLD_PRICE",
                disabled=False,
                required=True,
                min_value=0,
                step=0.01,
            ),
            "COVID_STRINGENCY_INDEX": st.column_config.NumberColumn(
                "COVID_STRINGENCY_INDEX",
                disabled=False,
                required=True,
                min_value=0,
                step=0.01,
            ),
            "UNRATE": st.column_config.NumberColumn(
                "UNRATE",
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
            "GDP": st.column_config.NumberColumn(
                "GDP",
                disabled=False,
                required=True,
                min_value=0,
                step=0.01,
            ),
        }
        cols = st.columns((0.5, 1))
        with cols[0]:
            if 'selected_exo_vars' not in st.session_state:
                st.session_state['selected_exo_vars'] = ['S&P500', 'GOLD_PRICE', 'CRUDE_PRICE', 'CPI']
            st.multiselect('Select variable to show', options=plot_show_df.drop('ds', axis=1).columns,
                           key='selected_exo_varss',
                           on_change=record_changes, args=('selected_exo_vars', 'selected_exo_varss'),
                           default=st.session_state['selected_exo_vars'])
            st.text("Edit Exogenous data:")
            edited_exo_data = st.data_editor(
                st.session_state["edit_exo_df"],
                key="edited_exo_data",
                hide_index=True,
                column_config=column_config,
                on_change=update_exo_df,
            )

        st.button('Save Edited Exogenous Data', key='save_exo_data')
        if st.session_state['save_exo_data']:
            st.session_state['edit_exo_df'] = edited_exo_data.copy()

        with cols[1]:
            scaled_exo_df = st.session_state['edit_exo_df'].copy()
            scaled_exo_df = scaled_exo_df.drop(columns=['ds'])
            scaler = MinMaxScaler()
            scaled_exo_df = pd.DataFrame(scaler.fit_transform(scaled_exo_df), columns=scaled_exo_df.columns)
            # scaled_exo_df = scaled_exo_df.rolling(window=5).mean()  # Apply rolling mean with window size 5
            scaled_exo_df = scaled_exo_df[st.session_state['selected_exo_vars']]
            scaled_exo_df['ds'] = st.session_state['edit_exo_df']['ds']
            fig = go.Figure()
            cols = scaled_exo_df.drop(columns=['ds']).columns
            for col in cols:
                fig.add_trace(
                    go.Scatter(
                        x=scaled_exo_df.ds,
                        y=scaled_exo_df[col],
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

    # if not st.session_state['is_train']:
    #     st.stop()
    st.session_state['user'] = 'test_all_1'
    if 'is_train' not in st.session_state:
        st.session_state['is_train'] = False
    if st.button('Train Models'):
        st.session_state['experiment_id'] = train_models(st.session_state['user'])
        st.session_state['is_train'] = True

    with st.expander('Future Forecasts', expanded=True):
        if 'fetch_forecast_data' not in st.session_state:
            st.session_state['fetch_forecast_data'] = fetch_forecast_data()
        fetched_data = st.session_state['fetch_forecast_data'].copy()
        fetched_data = fetched_data.rename(
            columns={'BUSINESS_UNIT': 'bu', 'DATE': 'ds', 'REVENUE': 'y', 'FORECAST_DATE': 'forecasted_ds',
                     'MODEL_NAME': 'model', "USER": "user", "EXPERIMENT_ID": "experiment_id"})
        fetched_data['y'] = fetched_data['y'].astype(float)
        fetched_data['ds'] = pd.to_datetime(fetched_data['ds'])
        fetched_data['forecasted_ds'] = pd.to_datetime(fetched_data['forecasted_ds'])
        cols = st.columns((1, 1, 1))
        if 'bu_forecast_selected' not in st.session_state:
            st.session_state['bu_forecast_selected'] = \
                (['Overall'] + st.session_state['all_bu_dfs']['bu'].unique().tolist())[2]
        cols[0].selectbox('Select Business Unit',
                          options=['Overall'] + st.session_state['all_bu_dfs']['bu'].unique().tolist(),
                          key='bu_forecast_selectedd',
                          index=(['Overall'] + st.session_state['all_bu_dfs']['bu'].unique().tolist()).index(
                              st.session_state['bu_forecast_selected']),
                          on_change=record_changes, args=('bu_forecast_selected', 'bu_forecast_selectedd'))
        if 'forecast_months' not in st.session_state:
            st.session_state['forecast_months'] = 12
        cols[1].selectbox('Select no of months to forecast', options=[3, 6, 9, 12], key='forecast_monthss',
                          index=[3, 6, 9, 12].index(st.session_state['forecast_months']),
                          on_change=record_changes, args=('forecast_months', 'forecast_monthss'))
        if 'selected_models' not in st.session_state:
            st.session_state['selected_models'] = fetched_data['experiment_id'].unique().tolist()
        cols[2].multiselect('Select models', options=fetched_data['experiment_id'].unique().tolist(),
                            key='selected_modelss',
                            default=st.session_state['selected_models'],
                            on_change=record_changes, args=('selected_models', 'selected_modelss'))

        if len(list(st.session_state['selected_models'])) == 0:
            st.write('Please select atleast one model')
            st.stop()

        forecast_date = pd.to_datetime(latest_actual_date_time) + relativedelta(
            months=+st.session_state['forecast_months'])
        # st.session_state['models'] = {}
        # st.session_state['models']['forecasted_dates'] = [pd.to_datetime('2024-08-13'), pd.to_datetime('2024-08-16')]
        # forecast_df = pd.DataFrame(columns=['bu', 'ds', 'y', 'model', 'forecasted_ds'])
        # for i, model in enumerate(st.session_state['selected_models']):
        #     st.session_state['models'][model] = pd.read_excel('data/models.xlsx', sheet_name=model)
        #     st.session_state['models'][model]['model'] = model
        #     st.session_state['models'][model]['forecasted_ds'] = st.session_state['models']['forecasted_dates'][i]
        #     st.session_state['models'][model]['forecasted_ds'] = st.session_state['models'][model]['forecasted_ds']
        #     forecast_df = pd.concat([forecast_df, st.session_state['models'][model]]).reset_index(drop=True)
        #     insert_forecast_into_db(st.session_state['models'][model], 'karthik', model)

        forecast_df = fetched_data.copy()
        forecast_df = forecast_df[(forecast_df['ds'] <= pd.to_datetime(forecast_date))]
        # forecast_df = forecast_df[(forecast_df['ds'] <= pd.to_datetime(forecast_date)) & (
        #         forecast_df['ds'] > pd.to_datetime(latest_actual_date_time))]
        forecast_df['ds'] = forecast_df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
        forecast_df['forecasted_ds'] = forecast_df['forecasted_ds'].dt.strftime('%Y-%m-%d')
        st.session_state['future_forecast'] = forecast_df.copy()

        fig = go.Figure()
        if st.session_state['bu_forecast_selected'] == "Overall":
            for bu in st.session_state['all_bu_dfs']['bu'].unique():
                bu_data_actual = st.session_state['all_bu_dfs'][st.session_state['all_bu_dfs']['bu'] == bu]
                bu_data_actual = bu_data_actual[
                    bu_data_actual['ds'] >= str(pd.to_datetime(latest_actual_date_time) + relativedelta(
                        years=-int(st.session_state['past_years'])))]
                fig.add_trace(
                    go.Scatter(
                        x=bu_data_actual['ds'],
                        y=bu_data_actual['y'],
                        mode='lines',
                        name=f"{bu}_Actual"
                    )
                )
                bu_data_forecast = forecast_df[forecast_df['bu'] == bu]
                bu_data_forecast = bu_data_forecast[
                    bu_data_forecast['ds'] >= str(pd.to_datetime(latest_actual_date_time) + relativedelta(
                        years=-int(st.session_state['past_years'])))]
                for model in st.session_state['selected_models']:
                    bu_data_model = bu_data_forecast[bu_data_forecast['experiment_id'] == model]
                    fig.add_trace(
                        go.Scatter(
                            x=bu_data_model['ds'],
                            y=bu_data_model['y'],
                            mode='lines',
                            line=dict(dash='dot'),
                            name=f"{bu}_{model}_Forecast"
                        )
                    )
        else:
            bu_data_actual = st.session_state['all_bu_dfs'][
                st.session_state['all_bu_dfs']['bu'] == st.session_state['bu_forecast_selected']]
            bu_data_actual = bu_data_actual[
                bu_data_actual['ds'] >= str(pd.to_datetime(latest_actual_date_time) + relativedelta(
                    years=-int(st.session_state['past_years'])))]
            fig.add_trace(
                go.Scatter(
                    x=bu_data_actual['ds'],
                    y=bu_data_actual['y'],
                    mode='lines',
                    name=f"{st.session_state['bu_forecast_selected']}_Actual"
                )
            )
            bu_data_forecast = forecast_df[forecast_df['bu'] == st.session_state['bu_forecast_selected']]
            for model in st.session_state['selected_models']:
                bu_data_model = bu_data_forecast[bu_data_forecast['experiment_id'] == model]
                fig.add_trace(
                    go.Scatter(
                        x=bu_data_model['ds'],
                        y=bu_data_model['y'],
                        mode='lines',
                        line=dict(dash='dot'),
                        name=f"{st.session_state['bu_forecast_selected']}_{model}_Forecast"
                    )
                )
        fig.update_layout(
            title=f"{st.session_state['bu_forecast_selected']} Business Units Forecast Graph",
            xaxis_title='Date',
            yaxis_title='Value',
            legend_title='Business Units'
        )

        st.plotly_chart(fig)
        if st.button('Refresh Forecast'):
            st.session_state['fetch_forecast_data'] = fetch_forecast_data()
            st.session_state['selected_models'] = st.session_state['fetch_forecast_data']['experiment_id'].unique().tolist()
        st.divider()

        st.header('Forecast Data')
        ## Download forecast data
        download_table = forecast_df.copy()
        download_table['ds'] = pd.to_datetime(download_table['ds'])
        download_table['ds'] = download_table['ds'].dt.date
        download_table['y'] = download_table['y'].astype(float)
        download_table = download_table.rename(
            columns={'ds': 'Date', 'y': 'Revenue', 'bu': 'Business Unit', 'forecasted_ds': 'Forecasted Date',
                     'model': 'Model', 'experiment_id': 'Experiment ID'})
        download_table = download_table[
            ['Business Unit', 'Date', 'Forecasted Date', 'Revenue', 'Model', 'Experiment ID']]
        download_table_filename = 'future-forecast.xlsx'
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            for i, model in enumerate(st.session_state['selected_models']):
                model_table = download_table[download_table['Experiment ID'] == model].reset_index(drop=True)
                model_table = model_table.drop(columns=['Model', 'Experiment ID'])
                model_table.to_excel(writer, index=False, sheet_name=f"Model_{i + 1}")
            writer.close()
        st.download_button(label="Download Data", data=buffer.getvalue(), file_name=download_table_filename,
                           mime="application/vnd.ms-excel", key='download_forecast_data')

        ## Display forecast data
        if st.session_state['bu_forecast_selected'] == "Overall":
            forecast_table = download_table.copy()
        else:
            forecast_table = download_table[
                download_table['Business Unit'] == st.session_state['bu_forecast_selected']].copy()
        forecast_table['Business Unit'] = forecast_table['Business Unit'].str.upper()
        for i, model in enumerate(st.session_state['selected_models']):
            model_table = forecast_table[forecast_table['Experiment ID'] == model].reset_index(drop=True)
            st.markdown(f"### Model {i + 1}")
            model_table = model_table.drop(columns=['Model', 'Experiment ID'])
            st.table(
                # model_table
                model_table.set_index("Business Unit")
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
                    ],
                    overwrite=False,
                )
                .format(precision=2)
            )
