from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
import pandas as pd
import snowflake.connector as connector
import os
import streamlit as st
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv
load_dotenv()
# # @st.cache_resource()
# def create_snowflake_connection(user, role):
#     return connector.connect(
#         account=os.environ["SNOWFLAKE_ACCOUNT"],
#         user=user,
#         authenticator="externalbrowser",
#         warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
#         role=role,
#     )

def execute_snowflake_query(sql_query):
    try:
        conn = create_snowflake_connection()
        cur = conn.cursor()
        res = cur.execute(sql_query)
        conn.commit()
        conn.close()
        return res.fetchone()[0]
    except Exception as e:
        print(e)
        pass
    finally:
        if conn:    
            conn.close()

def insert_data_to_snowflake(df, table_name):
    try:
        conn = create_snowflake_connection()
        write_pandas(conn, df, table_name, database=os.environ["SNOWFLAKE_DATABASE"], schema=os.environ["SNOWFLAKE_SCHEMA"])
    except Exception as e:
        st.error(e)
    finally:
        if conn:
            conn.close()

def create_snowflake_connection():
    con = connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse='POWERHOUSE',
    )
    return con


def convert_period_to_dates(date,duration):
    today = date or datetime.now().date()
    end_month = today - relativedelta(days=1)
    start_month = end_month - relativedelta(months=duration - 1)
    print(start_month, end_month)
    return start_month, end_month


def convert_dates_to_string(start_date, end_date):
    # end_date = end_date + relativedelta(months=1)
    return ",".join(
        [
            f"""'{date.strftime("%Y-%m-%d")}'"""
            for date in pd.date_range(start_date, end_date, freq="ME")
        ]
    )

def get_end_of_month(year, month):
    month_number = datetime.strptime(month, '%b').month
    last_day = calendar.monthrange(year, month_number)[1]
    last_date = datetime(year, month_number, last_day).date()
    # print(f"[INFO] Last date of the month: {last_date}")
    return last_date

@st.cache_data(show_spinner=True)
def fetch_data_from_db(dates, later=True):
    print("[INFO] fetching data from SNOWFLAKE")
    sql_query = f"""
    WITH cte AS
    (
        SELECT
            CASE
            WHEN deal_stage = '7- Closed Won' THEN '7_closed_won'
            WHEN deal_stage = 'Closed Won' THEN '7_closed_won'
            WHEN deal_stage = 'Closed won' THEN '7_closed_won'
            WHEN deal_stage = '7 - Closed Won' THEN '7_closed_won'
            WHEN deal_stage = '7- Closed Lost' THEN '7_closed_lost'
            WHEN deal_stage = 'Closed Dead' THEN '7_closed_lost'
            WHEN deal_stage = 'Closed Lost' THEN '7_closed_lost'
            WHEN deal_stage = 'Closed lost' THEN '7_closed_lost'
            WHEN deal_stage = '7 - Closed Dead' THEN '7_closed_lost'
            WHEN deal_stage = '7 - Closed Lost' THEN '7_closed_lost'
            WHEN deal_stage = '6- Contracting' THEN '6_contracting'
            WHEN deal_stage = 'Contracting' THEN '6_contracting'
            WHEN deal_stage = '5- Verbal Agreement' THEN '5_verbal_agreement'
            WHEN deal_stage = 'Verbal Agreement' THEN '5_verbal_agreement'
            WHEN deal_stage = '4- Proposal Presented' THEN '4_proposal_presented'
            WHEN deal_stage = '4 - Proposal ' THEN '4_proposal_presented'
            WHEN deal_stage = '3- Qualified Opportunity' THEN '3_qualified_oppurtunity'
            WHEN deal_stage = 'Qualified Opportunity' THEN '3_qualified_oppurtunity'
            WHEN deal_stage = '3 - Qualified Opportunity (NDA + RATE CARD sent)' THEN '3_qualified_oppurtunity'
            WHEN deal_stage = 'Proposal' THEN '4_proposal_presented'
            WHEN deal_stage = '2 - Needs Expressed' THEN '2_needs_expressed'
            WHEN deal_stage = '2- Needs Expressed' THEN '2_needs_expressed'
            WHEN deal_stage = 'Needs Expressed' THEN '2_needs_expressed'
            WHEN deal_stage = '1- Connected to Meet' THEN '1_connected_to_meet'
            WHEN deal_stage = '1 - Connected to Meet' THEN '1_connected_to_meet'
            WHEN deal_stage = 'Connected to Meet' THEN '1_connected_to_meet'
            WHEN deal_stage = '0- New' THEN '0_new'
            WHEN deal_stage = '0 - New ' THEN '0_new'
            WHEN deal_stage = 'Discovery' THEN '0_new'
            WHEN deal_stage = 'Business Considerations' THEN '0_new'
            WHEN deal_stage = 'Pricing and Terms' THEN '4_proposal_presented'
            WHEN deal_stage = 'Solution Demo' THEN '4_proposal_presented'
            END AS deal_stage,
            snapshot_date,
            record_id,
            deal_name,
            customer_segment,
            associated_company,
            deal_probability,
            AMOUNT_IN_COMPANY_CURRENCY,
            ENGAGEMENT_TYPE,
            FULFILLMENT_TYPE,
            EXPECTED_PROJECT_START_DATE,
            EXPECTED_PROJECT_DURATION_IN_MONTHS,
            GAAP,
            ANNUAL_CONTRACT_VALUE,
            HUBSPOT_CAMPAIGN,
            TOTAL_CONTRACT_VALUE,
            REVENUE_TYPE,
            PIPELINE,
            CLOSE_DATE,
            CREATE_DATE,
            DEAL_TYPE,
            qualification_date,
            work_ahead,
            effective_probability,
            "EST._PROJECT_END_DATE",
            "EST._MONTHLY_REVENUE_(COMPANY_CURRENCY)",
            original_deal_id,
            DATE_ENTERED_WORK_AHEAD,
            project_end_date,
            TCV_AND_AMOUNT_DELTA,
            DENSE_RANK() OVER (PARTITION BY TO_DATE(SNAPSHOT_DATE) ORDER BY TO_TIMESTAMP(SNAPSHOT_DATE) {'DESC' if later else 'ASC'}) AS rn
        FROM DSX_DASHBOARDS.HUBSPOT_RAW.DEAL_RAW
        WHERE CAST(SNAPSHOT_DATE AS DATE) in ({dates})
    )
    SELECT *
    FROM cte
    WHERE rn=1
    AND (PIPELINE like '%DSX%' OR PIPELINE like '%Renewals%')
    AND deal_stage NOT IN ('7_closed_won', '7_closed_lost')
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

@st.cache_data(show_spinner=False)
def fetch_weightages():
    print("[INFO] fetching weightages from SNOWFLAKE")
    sql_query = f"""
    WITH cte AS
    (
        SELECT
            CASE
            WHEN deal_stage = '7- Closed Won' THEN '7_closed_won'
            WHEN deal_stage = 'Closed Won' THEN '7_closed_won'
            WHEN deal_stage = 'Closed won' THEN '7_closed_won'
            WHEN deal_stage = '7 - Closed Won' THEN '7_closed_won'
            WHEN deal_stage = '7- Closed Lost' THEN '7_closed_lost'
            WHEN deal_stage = 'Closed Dead' THEN '7_closed_lost'
            WHEN deal_stage = 'Closed Lost' THEN '7_closed_lost'
            WHEN deal_stage = 'Closed lost' THEN '7_closed_lost'
            WHEN deal_stage = '7 - Closed Dead' THEN '7_closed_lost'
            WHEN deal_stage = '7 - Closed Lost' THEN '7_closed_lost'
            WHEN deal_stage = '6- Contracting' THEN '6_contracting'
            WHEN deal_stage = 'Contracting' THEN '6_contracting'
            WHEN deal_stage = '5- Verbal Agreement' THEN '5_verbal_agreement'
            WHEN deal_stage = 'Verbal Agreement' THEN '5_verbal_agreement'
            WHEN deal_stage = '4- Proposal Presented' THEN '4_proposal_presented'
            WHEN deal_stage = '4 - Proposal ' THEN '4_proposal_presented'
            WHEN deal_stage = '3- Qualified Opportunity' THEN '3_qualified_oppurtunity'
            WHEN deal_stage = 'Qualified Opportunity' THEN '3_qualified_oppurtunity'
            WHEN deal_stage = '3 - Qualified Opportunity (NDA + RATE CARD sent)' THEN '3_qualified_oppurtunity'
            WHEN deal_stage = 'Proposal' THEN '4_proposal_presented'
            WHEN deal_stage = '2 - Needs Expressed' THEN '2_needs_expressed'
            WHEN deal_stage = '2- Needs Expressed' THEN '2_needs_expressed'
            WHEN deal_stage = 'Needs Expressed' THEN '2_needs_expressed'
            WHEN deal_stage = '1- Connected to Meet' THEN '1_connected_to_meet'
            WHEN deal_stage = '1 - Connected to Meet' THEN '1_connected_to_meet'
            WHEN deal_stage = 'Connected to Meet' THEN '1_connected_to_meet'
            WHEN deal_stage = '0- New' THEN '0_new'
            WHEN deal_stage = '0 - New ' THEN '0_new'
            WHEN deal_stage = 'Discovery' THEN '0_new'
            WHEN deal_stage = 'Business Considerations' THEN '0_new'
            WHEN deal_stage = 'Pricing and Terms' THEN '4_proposal_presented'
            WHEN deal_stage = 'Solution Demo' THEN '4_proposal_presented'
            END AS deal_stage,
            weightage from DSX_DASHBOARDS.HUBSPOT_RAW.WEIGHTAGE)
        SELECT * from cte WHERE deal_stage NOT IN ('7_closed_won', '7_closed_lost')
        """.strip()

    try:
        conn = create_snowflake_connection()
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
    return None
