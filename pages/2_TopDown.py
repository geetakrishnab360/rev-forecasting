import streamlit as st
import plotly.graph_objs as go
from io import BytesIO
from holoviews.plotting.bokeh.styles import font_size
from jupyterlab.semver import valid_range
from plotly.subplots import make_subplots
from plotly import colors
from dotenv import load_dotenv
from scipy.optimize import minimize
from datetime import datetime
import datetime as datet
from dateutil.relativedelta import relativedelta
from snowflake_utils import (
    convert_period_to_dates,
    convert_dates_to_string,
    fetch_data_from_db,
    fetch_weightages,
)
import pandas as pd
import numpy as np
from data_utils import (
    preprocess,
    calculate_forecasts,
    _calculate_snapshot_monthly_number,
    calculate_cohort_error,
    convert_to_cohort_df,
    prepare_actual_rev_data,
    aggregate_snapshot_numbers,
    create_exo_df,
    preprocess_bu_rev,
    prepare_bu_revenue_data,
    update_df,
)
from db import (
    create_database,
    insert_experiment_into_db,
    fetch_all_experiments,
    fetch_experiment,
    create_bu_database,
    insert_bu_into_db,
    delete_bu_from_db,
    fetch_all_bu_data,
    update_bu_in_db,
    create_forecast_database,
    insert_forecast_into_db,
    fetch_all_forecast_data,
    delete_forecast_from_db,
)
from components import (
    set_header,
    hide_sidebar,
)
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Top Down", initial_sidebar_state="collapsed")
hide_sidebar()
page = option_menu(
    menu_title=None,
    options=["Pipeline Analysis", "Pipeline Forecast", "TopDown Forecast"],
    default_index=2,
    orientation="horizontal",
)
if page == 'Pipeline Analysis':
    st.switch_page("Analysis.py")
if page == 'Pipeline Forecast':
    st.switch_page("pages/1_Forecast_Results.py")
set_header("Top Down Forecasting")

load_dotenv()
create_bu_database()
create_forecast_database()
delete_forecast_from_db('karthik')

with open("./styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )

if "bu_actual_dfs" not in st.session_state:
    st.session_state["bu_actual_dfs"] = preprocess_bu_rev(prepare_bu_revenue_data())

if 'is_bu_inserted' not in st.session_state:
    delete_bu_from_db('karthik')
    insert_bu_into_db(st.session_state["bu_actual_dfs"], 'karthik')
    st.session_state['is_bu_inserted'] = True

if 'is_fetched' not in st.session_state and 'is_bu_inserted' in st.session_state and st.session_state['is_bu_inserted']:
    st.session_state['all_bu_dfs'] = fetch_all_bu_data('karthik')
    st.session_state['is_fetched'] = True

latest_actual_date_time = st.session_state['all_bu_dfs']['ds'].max()
st.session_state['latest_actual_date_time'] = latest_actual_date_time
latest_actual_date = datet.datetime.strptime(latest_actual_date_time,
                                             "%Y-%m-%d %H:%M:%S").date() + relativedelta(months=+1)
st.session_state['latest_actual_date'] = latest_actual_date

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

        # selected_data = st.session_state['edit_bu'][st.session_state['edit_bu']['selected']]
        # selected_data = selected_data.rename(columns={'BU': 'bu', 'Date': 'ds', 'Revenue': 'y'})
        # selected_data['y'] = selected_data['y'].astype(float)
        # selected_data['ds'] = selected_data['ds'].apply(lambda x: x.replace(day=1)).dt.strftime('%Y-%m-%d %H:%M:%S')
        # st.session_state['is_update'] = False
        # if len(selected_data) != 0:
        #     selected_data = selected_data[['bu', 'ds', 'y']]
        #     for i, row in selected_data.iterrows():
        #         if len(st.session_state['all_bu_dfs'].loc[(st.session_state['all_bu_dfs']['bu'] == row['bu']) & (
        #                 st.session_state['all_bu_dfs']['ds'] == row['ds'])]) != 0:
        #             st.info("Are you sure you want to update the data?")
        #             st.session_state['is_update'] = not st.button('Yes')
        #             break
        #
        # if st.button('Save Edited Data', disabled=st.session_state['is_update']):
        #     selected_data = st.session_state['edit_bu'][st.session_state['edit_bu']['selected']]
        #     selected_data = selected_data.rename(columns={'BU': 'bu', 'Date': 'ds', 'Revenue': 'y'})
        #     selected_data['y'] = selected_data['y'].astype(float)
        #     selected_data['ds'] = selected_data['ds'].apply(lambda x: x.replace(day=1)).dt.strftime('%Y-%m-%d %H:%M:%S')
        #
        #     if len(selected_data) != 0:
        #         selected_data = selected_data[['bu', 'ds', 'y']]
        #         for i, row in selected_data.iterrows():
        #             if len(st.session_state['all_bu_dfs'].loc[(st.session_state['all_bu_dfs']['bu'] == row['bu']) & (
        #                     st.session_state['all_bu_dfs']['ds'] == row['ds'])]) != 0:
        #                 row_c = pd.DataFrame(row).T
        #                 row_c = row_c.reset_index(drop=True)
        #                 update_bu_in_db(row_c, 'karthik')
        #                 st.session_state['all_bu_dfs'].loc[(st.session_state['all_bu_dfs']['bu'] == row['bu']) & (
        #                         st.session_state['all_bu_dfs']['ds'] == row['ds']), 'y'] = row['y']
        #             else:
        #                 row_c = pd.DataFrame(row).T
        #                 row_c = row_c.reset_index(drop=True)
        #                 insert_bu_into_db(row_c, 'karthik')
        #                 st.session_state['all_bu_dfs'] = pd.concat([st.session_state['all_bu_dfs'], row_c])
        #                 st.session_state['all_bu_dfs'] = st.session_state['all_bu_dfs'].reset_index(drop=True)

        # new
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
                            update_bu_in_db(row_c, 'karthik')
                            st.session_state['all_bu_dfs'].loc[
                                (st.session_state['all_bu_dfs']['bu'] == row['bu']) &
                                (st.session_state['all_bu_dfs']['ds'] == row['ds']), 'y'] = row['y']
                        else:
                            insert_bu_into_db(row_c, 'karthik')
                            st.session_state['all_bu_dfs'] = pd.concat(
                                [st.session_state['all_bu_dfs'], row_c]).reset_index(
                                drop=True)
                    cols = st.columns([0.4, 1])
                    cols[0].success("Data saved successfully")
        # new end

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
        cols[0].selectbox('Select Business Unit',
                          options=['Overall'] + st.session_state['all_bu_dfs']['bu'].unique().tolist(),
                          key='bu_graph_selected', index=0)
        no_of_past_years = latest_actual_date + relativedelta(months=-1) - datet.datetime.strptime(
            st.session_state['all_bu_dfs']['ds'].min(), "%Y-%m-%d %H:%M:%S").date()
        # print(no_of_past_years)
        no_of_past_years = no_of_past_years.days // 365
        # print(datet.datetime.strptime(st.session_state['all_bu_dfs']['ds'].min(), "%Y-%m-%d %H:%M:%S").date())
        # print(latest_actual_date)
        cols[1].selectbox('Select past years of actuals', options=range(1, no_of_past_years + 1), key='past_years',
                          index=1)
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
            title=f'{st.session_state['bu_graph_selected']} Business Units Graph',
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
            st.multiselect('Select variable to show', options=plot_show_df.columns, key='selected_exo_vars',
                           default=['S&P500', 'GOLD_PRICE', 'CRUDE_PRICE', 'CPI'])
            st.text("Edit Exogenous data:")
            edited_exo_data = st.data_editor(
                st.session_state["edit_exo_df"],
                key="edited_exo_data",
                hide_index=True,
                column_config=column_config,
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
        st.button('Run Models', key='run_models')

    with st.expander('Future Forecasts', expanded=True):
        cols = st.columns((1, 1, 1))
        cols[0].selectbox('Select Business Unit',
                          options=['Overall'] + st.session_state['all_bu_dfs']['bu'].unique().tolist(),
                          key='bu_forecast_selected', index=2)
        cols[1].selectbox('Select no of months to forecast', options=[3, 6, 9, 12], key='forecast_months', index=3)
        cols[2].multiselect('Select models', options=['model1', 'model2'], key='selected_models',
                            default=['model1', 'model2'])

        if len(list(st.session_state['selected_models'])) == 0:
            st.write('Please select atleast one model')
            st.stop()
        # with cols[1]:
        #     dates_full = [(st.session_state['latest_actual_date'] + relativedelta(months=+i)) for i in range(12)]
        #     years = sorted(set(dates_full[i].strftime("%Y") for i in range(12)))
        #     forecast_year = st.selectbox('Select Year', options=years, index=len(years) - 1)
        # with cols[2]:
        #     valid_months = [date.strftime("%B") for date in dates_full if date.year == int(forecast_year)]
        #     forecast_month = st.selectbox('Select Month', options=valid_months, index=len(valid_months) - 1)
        # forecast_date = pd.to_datetime(f'{forecast_year}-{forecast_month}-01')
        forecast_date = pd.to_datetime(latest_actual_date_time) + relativedelta(
            months=+st.session_state['forecast_months'])
        # forecasted_date = pd.to_datetime(latest_actual_date_time)
        st.session_state['models'] = {}
        st.session_state['models']['forecasted_dates'] = [pd.to_datetime('2024-08-13'), pd.to_datetime('2024-08-16')]
        forecast_df = pd.DataFrame(columns=['bu', 'ds', 'y', 'model', 'forecasted_ds'])
        for i, model in enumerate(st.session_state['selected_models']):
            st.session_state['models'][model] = pd.read_excel('data/models.xlsx', sheet_name=model)
            st.session_state['models'][model]['model'] = model
            st.session_state['models'][model]['forecasted_ds'] = st.session_state['models']['forecasted_dates'][i]
            st.session_state['models'][model]['forecasted_ds'] = st.session_state['models'][model]['forecasted_ds']
            forecast_df = pd.concat([forecast_df, st.session_state['models'][model]]).reset_index(drop=True)
            insert_forecast_into_db(st.session_state['models'][model], 'karthik', model)

        fetched_data = fetch_all_forecast_data('karthik')
        fetched_data['ds'] = pd.to_datetime(fetched_data['ds'])
        fetched_data['forecasted_ds'] = pd.to_datetime(fetched_data['forecasted_ds'])
        forecast_df = fetched_data.copy()

        forecast_df = forecast_df[(forecast_df['ds'] <= pd.to_datetime(forecast_date)) & (
                forecast_df['ds'] > pd.to_datetime(latest_actual_date_time))]
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
                    bu_data_model = bu_data_forecast[bu_data_forecast['model'] == model]
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
                bu_data_model = bu_data_forecast[bu_data_forecast['model'] == model]
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
            title=f'{st.session_state['bu_forecast_selected']} Business Units Forecast Graph',
            xaxis_title='Date',
            yaxis_title='Value',
            legend_title='Business Units'
        )

        st.plotly_chart(fig)

        st.divider()

        st.header('Forecast Data')
        ## Download forecast data
        download_table = forecast_df.copy()
        download_table['ds'] = pd.to_datetime(download_table['ds'])
        download_table['ds'] = download_table['ds'].dt.date
        download_table['y'] = download_table['y'].astype(float)
        download_table = download_table.rename(
            columns={'ds': 'Date', 'y': 'Revenue', 'bu': 'BU', 'forecasted_ds': 'Forecasted Date', 'model': 'Model'})
        download_table_filename = 'future-forecast.xlsx'
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            for i, model in enumerate(st.session_state['selected_models']):
                model_table = download_table[download_table['Model'] == model].reset_index(drop=True)
                model_table = model_table.drop(columns=['Model'])
                model_table.to_excel(writer, index=False, sheet_name=f"Model_{i + 1}")
            writer.close()
        st.download_button(label="Download Data", data=buffer.getvalue(), file_name=download_table_filename,
                           mime="application/vnd.ms-excel", key='download_forecast_data')

        ## Display forecast data
        if st.session_state['bu_forecast_selected'] == "Overall":
            forecast_table = st.session_state['future_forecast'].copy()
        else:
            forecast_table = forecast_df[forecast_df['bu'] == st.session_state['bu_forecast_selected']].copy()
        forecast_table['ds'] = pd.to_datetime(forecast_table['ds'])
        forecast_table['ds'] = forecast_table['ds'].dt.date
        forecast_table['y'] = forecast_table['y'].astype(float)
        forecast_table = forecast_table[['bu', 'ds', 'forecasted_ds', 'y', 'model']]
        forecast_table = forecast_table.rename(
            columns={'ds': 'Date', 'y': 'Revenue', 'bu': 'BU', 'forecasted_ds': 'Forecasted Date', 'model': 'Model'})
        for i, model in enumerate(st.session_state['selected_models']):
            model_table = forecast_table[forecast_table['Model'] == model].reset_index(drop=True)
            st.markdown(f"### Model {i + 1}")
            model_table = model_table.drop(columns=['Model'])
            st.table(model_table
                     # model_table.set_index("BU")
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

        # st.table(
        #     forecast_table.set_index("BU")
        #     .style.set_table_styles(
        #         [
        #             {
        #                 "selector": "thead  th",
        #                 # fmt: off
        #                 "props": "background-color: #2d7dce; text-align:center; color: white;font-size:0.9rem;border-bottom: 1px solid black !important;",
        #                 # fmt: on
        #             },
        #             {
        #                 "selector": "tbody  th",
        #                 "props": "font-weight:bold;font-size:0.9rem;color:#000;",
        #             },
        #         ],
        #         overwrite=False,
        #     )
        #     .format(precision=2)
        # )
        # actual_forecast_df = pd.concat([st.session_state['all_bu_dfs'], forecast_df]).reset_index(drop=True)
        # st.session_state['actual_forecast_df'] = actual_forecast_df.copy()
        #
        # fig = go.Figure()
        # if st.session_state['bu_forecast_selected'] == "Overall":
        #     for bu in actual_forecast_df['bu'].unique():
        #         bu_data = actual_forecast_df[actual_forecast_df['bu'] == bu]
        #
        #         bu_data_past = bu_data[bu_data['ds'] <= st.session_state['latest_actual_date_time']]
        #         fig.add_trace(
        #             go.Scatter(
        #                 x=bu_data_past['ds'],
        #                 y=bu_data_past['y'],
        #                 mode='lines',
        #                 name=f"{bu}_Actual"
        #             )
        #         )
        #         bu_data_future = bu_data[bu_data['ds'] > st.session_state['latest_actual_date_time']]
        #         fig.add_trace(
        #             go.Scatter(
        #                 x=bu_data_future['ds'],
        #                 y=bu_data_future['y'],
        #                 mode='lines',
        #                 line=dict(dash='dot'),
        #                 name=f"{bu}_Forecast"
        #             )
        #         )
        # else:
        #     bu_data = actual_forecast_df[actual_forecast_df['bu'] == st.session_state['bu_forecast_selected']]
        #     bu_data_past = bu_data[bu_data['ds'] <= st.session_state['latest_actual_date_time']]
        #     fig.add_trace(
        #         go.Scatter(
        #             x=bu_data_past['ds'],
        #             y=bu_data_past['y'],
        #             mode='lines',
        #             name=f"{st.session_state['bu_forecast_selected']}_Actual"
        #         )
        #     )
        #     bu_data_future = bu_data[bu_data['ds'] > st.session_state['latest_actual_date_time']]
        #     fig.add_trace(
        #         go.Scatter(
        #             x=bu_data_future['ds'],
        #             y=bu_data_future['y'],
        #             mode='lines',
        #             line=dict(dash='dot'),
        #             name=f"{st.session_state['bu_forecast_selected']}_Forecast"
        #         )
        #     )
        # fig.update_layout(
        #     title=f'{st.session_state['bu_forecast_selected']} Business Units Forecast Graph',
        #     xaxis_title='Date',
        #     yaxis_title='Value',
        #     legend_title='Business Units'
        # )
        #
        # st.plotly_chart(fig)
        #
        # st.divider()
        #
        # st.header('Forecast Data')
        # if st.session_state['bu_forecast_selected'] == "Overall":
        #     forecast_table = st.session_state['future_forecast'].copy()
        # else:
        #     forecast_table = bu_data_future.copy()
        # forecast_table['ds'] = pd.to_datetime(forecast_table['ds'])
        # forecast_table['ds'] = forecast_table['ds'].dt.date
        # forecast_table['y'] = forecast_table['y'].astype(float)
        # forecast_table['forecasted_ds'] = pd.to_datetime(forecasted_date)
        # forecast_table['forecasted_ds'] = forecast_table['forecasted_ds'].dt.date
        #
        # forecast_table = forecast_table.rename(
        #     columns={'ds': 'Date', 'y': 'Revenue', 'bu': 'BU', 'forecasted_ds': 'Forecasted Date'})
        # st.table(
        #     forecast_table.set_index("BU")
        #     .style.set_table_styles(
        #         [
        #             {
        #                 "selector": "thead  th",
        #                 # fmt: off
        #                 "props": "background-color: #2d7dce; text-align:center; color: white;font-size:0.9rem;border-bottom: 1px solid black !important;",
        #                 # fmt: on
        #             },
        #             {
        #                 "selector": "tbody  th",
        #                 "props": "font-weight:bold;font-size:0.9rem;color:#000;",
        #             },
        #         ],
        #         overwrite=False,
        #     )
        #     .format(precision=2)
        # )
