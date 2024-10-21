import snowflake.connector as connector
import pandas as pd
import streamlit as st

def create_snowflake_connection():
    conn = connector.connect(
        user='forecasting_app_service_account',
        account='c2gpartners.us-east-1',
        password='B!_ForecastApp2024',
        warehouse='POWERHOUSE',
    )
    return conn


@st.cache_data(show_spinner=False)
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


def insert_revenue_data_to_db(df, bu):
    try:
        conn = create_snowflake_connection()
        cur = conn.cursor()
        sql_query = f"""
        INSERT INTO DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.REVENUE_DATA VALUES
        """
        values = []
        for i in range(len(df)):
            values.append(
                f"('{bu}', '{df.iloc[i]['date']}', {df.iloc[i]['revenue']})"
            )
        sql_query += ", ".join(values)
        cur.execute(sql_query)
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)
        pass


def insert_experiments_data_to_db(dictionary):
    try:
        conn = create_snowflake_connection()
        cur = conn.cursor()
        sql_query = f"""
        INSERT INTO DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.EXPERIEMENTS
        (EXPERIMENT_ID, 
        BUSINESS_UNIT, 
        START_DATE, 
        STATUS, 
        USER, 
        FAILURE_REASON)
        VALUES
        ('{dictionary['experiment_id']}',
        '{dictionary['business_unit']}', 
        '{dictionary['start_date']}', 
        '{dictionary['status']}', 
        '{dictionary['user']}', 
        '{dictionary['failure_reason']}')
        """
        cur.execute(sql_query)
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)
        pass


def update_experiments_data_to_db(exp_id):
    try:
        conn = create_snowflake_connection()
        cur = conn.cursor()
        sql_query = f"""
        UPDATE DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.EXPERIEMENTS
        SET STATUS = 'completed', END_DATE = CURRENT_TIMESTAMP()
        WHERE EXPERIMENT_ID = exp_id
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
        INSERT INTO DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.MODEL_METRICS
        (EXPERIMENT_ID, 
        BUSINESS_UNIT, 
        USER, 
        FEATURES, 
        HYPERPARAMETERS, 
        FINAL_SCORE, 
        AVG_TRAIN_MAPE, 
        AVG_TRAIN_MAPE_DEV,
        AVG_TRAIN_CONTRIBUTION_DEV,
        AVG_TEST_MAPE,
        AVG_TEST_MAPE_DEV,
        AVG_GROWTH_ERROR)
        VALUES
        ('{request_id}', 
        '{bu}', 
        '{user}', 
        '{dictionary['features']}', 
        '{dictionary['yo']}',
        '{dictionary['final_score']}',
        '{dictionary['avg_train_mape']}', 
        '{dictionary['avg_train_mape_dev']}',
        '{dictionary['avg_train_contribution_dev']}',
        '{dictionary['avg_test_mape']}',
        '{dictionary['avg_test_mape_dev']}',
        '{dictionary['avg_growth_error']}')
        """
        cur.execute(sql_query)
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)
        pass


def update_rev_data(new_df, user):
    try:
        conn = create_snowflake_connection()
        cur = conn.cursor()

        for i, row in new_df.iterrows():
            sql_query = f"""
                UPDATE DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.REVENUE_DATA
                SET REVENUE = {row['revenue']}, 
                WHERE BUSINESS_UNIT = '{row['bu']}' AND DATE = '{row['ds']}'
            """
            cur.execute(sql_query)

        conn.commit()
        conn.close()
    except Exception as e:
        print(e)
        pass


def insert_new_rev_data(all_bu_dfs, user):
    try:
        conn = create_snowflake_connection()
        cur = conn.cursor()

        sql_query = f"""
            INSERT INTO DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.REVENUE_DATA (BUSINESS_UNIT, DATE, REVENUE)
            VALUES
        """
        values = []
        for i, row in all_bu_dfs.iterrows():
            values.append(
                f"('{row['bu']}', '{row['ds']}', {row['y']})"
            )
        sql_query += ", ".join(values)
        cur.execute(sql_query)
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)
        pass


# @st.cache_data(show_spinner=False)
def fetch_forecast_data():
    print("[INFO] fetching data from SNOWFLAKE")
    try:
        conn = create_snowflake_connection()
        cur = conn.cursor()
        sql_query = f"""
        SELECT
            * 
        FROM DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.FORECASTS
        """.strip()
        cur.execute(sql_query)
        data = pd.DataFrame(
            cur.fetchall(), columns=[desc[0] for desc in cur.description]
        )
        conn.close()
        return data
    except Exception as e:
        print(e)
        pass
