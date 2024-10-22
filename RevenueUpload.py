from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from streamlit_option_menu import option_menu
from components import set_header, hide_sidebar
from revenue_upload_utils import (update_client_list, 
                                  export_txn_df_to_excel,
                                  read_file, 
                                  read_client_list,
                                  preprocess,
                                  SUBSIDIARY_GROUPS)
from snowflake_utils import insert_data_to_snowflake,execute_snowflake_query

FILE_NAMES = ['netsuite revenue.xlsx', 'clients.xlsb']
DATA_PATH = Path("./")

def validate_the_uploaded_files(files):
    errors = []
    if len(files) < 2:
        errors.append(f"Please upload {[f.upper() for f in FILE_NAMES]} files")
    if len(files) >=2:
        for file in FILE_NAMES:
            if file.upper() not in [f.name.upper() for f in files]:
                errors.append(f"{file.upper()} file missing")
    if len(errors) == 0:
        for file in files:
            if file.name.lower() == FILE_NAMES[0]:
                try:
                    txn_df = read_file(file, sheet_name='Original', header_row=6)
                except ValueError as e:
                    if str(e).startswith("Worksheet named"):
                        errors.append("The uploaded netsuite file doesnt have a sheet named 'Original'")
            if file.name.lower() == FILE_NAMES[1]:
                try:
                    client_list = read_client_list(file)
                except Exception as e:
                    errors.append(f"The uploaded netsuite file is not a valid excel file. {e}")
    if len(errors) > 0:
        with upload_status_placeholder:
            st.error("\n".join(errors))
            return False, None, None
    else:
        with upload_status_placeholder:
            st.success("Upload Success")
    
    return True, txn_df, client_list
    
def get_stats(df):
    accounting_date = df['Accounting Period: Start Date'].dropna().unique()[0]
    num_clients = df.client.nunique()
    num_projects = df.project.nunique()
    percent_client_missing = (df.client=='NA').sum() / len(df) * 100
    percent_project_missing = (df.project=='NA').sum() / len(df) * 100
    percent_ns_missing = df["Entity (Line): Internal ID"].isna().sum() / len(df) * 100
    percent_hs_missing = df["Entity (Line): HubSpot Deal ID"].isna().sum() / len(df) * 100
    return pd.DataFrame([
        ("Accounting Date", accounting_date.strftime("%b, %Y")),
        ("# Clients", num_clients),
        ("# Projects", num_projects),
        ("% Txn missing Client", percent_client_missing),
        ("% Txn missing Project", percent_project_missing),
        ("% Txn missing NSID", percent_ns_missing),
        ("% Txn missing Hubspot ID", percent_hs_missing),
        
    ], columns = ['Metric', 'Value'])

def generate_revenue_report(df):
    mod_txn_df = df.copy()
    mod_txn_df = mod_txn_df[mod_txn_df["Subsidiary: Name"].isin([k for k,v in SUBSIDIARY_GROUPS.items() if v=='group1'])].copy()
    mod_txn_df = mod_txn_df[~mod_txn_df["Department: Name"].isin(['Insourcing Services','FPAI -General' ])].copy()
    mod_txn_df["Department: Name"] = mod_txn_df["Department: Name"].replace(
    {
        "Data Science & Insights": "DS",
        "Data Science": "DS",
        "Data Engineering": "DE",
        "Business Intelligence": "BI",
    }
    )

    revenue_df = (
        mod_txn_df.groupby(
            [
                "Entity (Line): Internal ID",
                "client",
                "project",
                "Department: Name",
                "Subsidiary: Name",
                "Accounting Period: Start Date",
            ]
        )["Amount"]
        .sum()
        .reset_index()
    )

    revenue_df["DEPARTMENT"] = ""
    revenue_df.rename(
        columns={
            "Entity (Line): Internal ID": "NETSUITE_PROJECT_ID",
            "client": "CLIENT",
            "project": "PROJECT",
            "Department: Name": "SSL",
            "Subsidiary: Name": "SUBSIDIARY",
            "Accounting Period: Start Date": "DATE",
            "Amount": "REVENUE",
        }
    )[
        [
            "NETSUITE_PROJECT_ID",
            "CLIENT",
            "PROJECT",
            "SSL",
            "SUBSIDIARY",
            "DEPARTMENT",
            "DATE",
            "REVENUE",
        ]
    ].to_csv("revenue1.csv", index=False)

def insert_revenue_report(df):
    mod_txn_df = df.copy()
    mod_txn_df = mod_txn_df[mod_txn_df["Subsidiary: Name"].isin([k for k,v in SUBSIDIARY_GROUPS.items() if v=='group1'])].copy()
    mod_txn_df = mod_txn_df[~mod_txn_df["Department: Name"].isin(['Insourcing Services','FPAI -General' ])].copy()
    mod_txn_df["Department: Name"] = mod_txn_df["Department: Name"].replace(
    {
        "Data Science & Insights": "DS",
        "Data Science": "DS",
        "Data Engineering": "DE",
        "Business Intelligence": "BI",
    }
    )

    revenue_df = (
        mod_txn_df.groupby(
            [
                "Entity (Line): Internal ID",
                "client",
                "project",
                "Department: Name",
                "Subsidiary: Name",
                "Accounting Period: Start Date",
            ]
        )["Amount"]
        .sum()
        .reset_index()
    )

    revenue_df["DEPARTMENT"] = ""
    revenue_df["Accounting Period: Start Date"] = revenue_df["Accounting Period: Start Date"].dt.strftime("%Y-%m-%d")
    revenue_df = revenue_df.rename(
        columns={
            "Entity (Line): Internal ID": "NETSUITE_PROJECT_ID",
            "client": "CLIENT",
            "project": "PROJECT",
            "Department: Name": "SSL",
            "Subsidiary: Name": "SUBSIDIARY",
            "Accounting Period: Start Date": "DATE",
            "Amount": "REVENUE",
        }
    )[
        [
            "NETSUITE_PROJECT_ID",
            "CLIENT",
            "PROJECT",
            "SSL",
            "SUBSIDIARY",
            "DEPARTMENT",
            "DATE",
            "REVENUE",
        ]
    ]
    insert_date = revenue_df.DATE.unique()[0]
    with cols[0]:
        st.success(f"Inserted data to Snowflake for {insert_date}")
    # execute_snowflake_query(f"""
    #                         DELETE FROM DSX_DASHBOARDS_SANDBOX.FORECASTING_TOOL.actual_revenues
    #                         WHERE DATE = '{insert_date}'
    #                         """)
    # insert_data_to_snowflake(revenue_df, "actual_revenues")
    
st.set_page_config(layout="wide", page_title="Top Down", initial_sidebar_state="collapsed")
hide_sidebar()
page = option_menu(
    menu_title=None,
    options=["Revenue Upload", "Pipeline Analysis", "Pipeline Forecast", "TopDown Forecast"],
    default_index=0,
    orientation="horizontal",
    icons=["database", "bar-chart-line-fill", "graph-up-arrow", "graph-up-arrow"],
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
    st.switch_page("pages/1_Analysis.py")
if page == 'Pipeline Forecast':
    st.switch_page("pages/2_ForecastResults.py")
if page == 'TopDown Forecast':
    st.switch_page("pages/3_TopDown.py")
set_header("Revenue Upload")

load_dotenv()

with open("./styles1.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )
    
if "txn_df" not in st.session_state:
    st.session_state['txn_df'] = None
    if Path("data/txn_df.csv").exists():
        st.session_state['txn_df'] = pd.read_csv("data/txn_df.csv")      
        st.session_state['txn_df']["Accounting Period: Start Date"] = pd.to_datetime(st.session_state['txn_df']["Accounting Period: Start Date"])
if "client_list" not in st.session_state:
    st.session_state['client_list'] = None
    if Path("data/client_list.txt").exists():
        with open("data/client_list.txt", 'r') as f:  
            st.session_state['client_list'] = f.readlines() 

if "is_valid" not in st.session_state:
    st.session_state['is_valid'] = False       
      
if st.session_state['txn_df'] is not None and st.session_state['client_list'] is not None:
    st.session_state['is_valid'] = True

cols = st.columns((0.4,0.1,0.5))
with cols[0]:
    st.checkbox(
        label="I have the final file",
        key='has_final_file'
    )
button_cols = st.columns(9)
if st.session_state['has_final_file']:
    with cols[0]:
        st.file_uploader("Upload Final file", type="xlsx", key='uploaded_final_file')
        if st.session_state['uploaded_final_file']:
            st.session_state['final_txn_df'] = pd.read_excel(st.session_state['uploaded_final_file'], sheet_name=None)
            
            if isinstance(st.session_state['final_txn_df'], dict):
                st.session_state['final_txn_df'] = st.session_state['final_txn_df'].get('Modified')
            
            # generate_revenue_report(st.session_state['final_txn_df'])
            st.success("Upload Success")

with cols[-1]:
    st.markdown("""
                <div class='instructions color1'>
                <h2>Instructions for file upload:</h2>
                <p>The upload should contain two files</p>
                <ol>
                <li>Netsuite revenue report Excel. Please name the sheet as 'Original'</li>
                <li>Netsuite report with Company names as Excel.</li>
                </ol>
                </div>
                """.strip(), unsafe_allow_html=True)
    st.markdown("""
                <div class='instructions color2'>
                <h2>How to use this Page?</h2>
                <p>This page will analyze the raw revenue report from Netsuite and create a project wise revenue report.</p>
                <p>Generated report contains:</p>
                <ol>
                <li>A 'Modified' sheet with client,project, hubspot id and netsuite id inferred from the original report.</li>
                <li>Client wise pivot table with total revenue for each client.</li>
                <li>Project wise pivot table with total revenue for each project.</li>
                <li>Client Rename Mapping - used to rename the clients found</li>
                <li>Clients list - Used for inferring clients from the Netsuite raw report</li>
                </ol>
                </div>
                """.strip(), unsafe_allow_html=True)
    st.markdown("""
                <div class='instructions color3'>
                <h2>What to do?</h2>
                <ol>
                <li>Verify the details from the Modified sheet and confirm the inferred clients and projects.</li>
                <li>Use Match score (a proxy for confidence on inferred) for navigating through the data.</li>
                <li>Verify netsuite id and hubspot deal ids.</li>
                <li>Please make any necessary changes and re-upload the file</li>
                <li>Re-uploaded file will be used to calculate netsuite id wise revenues.</li>
                </ol>
                </div>
                """.strip(), unsafe_allow_html=True)

if not st.session_state['has_final_file']:
    with cols[0]:
        st.file_uploader("Upload Raw files", type=["xlsx", "xlsb"], accept_multiple_files=True, key='uploaded_raw_files')
        upload_status_placeholder = st.container()
    
      
if st.session_state.get('uploaded_raw_files',None): 
    st.session_state['is_valid'], st.session_state['txn_df'], st.session_state['client_list'] = validate_the_uploaded_files(st.session_state['uploaded_raw_files'])
    st.session_state['uploaded'] = True

with cols[0]:
    if st.session_state['is_valid']:
        if st.session_state['txn_df']['Accounting Period: Start Date'].dropna().nunique() > 1:
            with upload_status_placeholder:
                st.error("More than one accounting period found. Please upload a single accounting period file.")
                st.stop()
        if st.session_state.get('mod_txn_df') is None:
            with upload_status_placeholder:
                with st.spinner("Analyzing data..."):
                    st.session_state['client_list'] = update_client_list(st.session_state['client_list'])
                    for c in st.session_state['client_list']:
                        if "point" in c.lower():
                            print(c)
                    st.session_state['mod_txn_df'], st.session_state['intercompany_txn'] = preprocess(st.session_state['txn_df'], st.session_state['client_list'])
                    final_txn_df = pd.concat([st.session_state['mod_txn_df'], st.session_state['mod_txn_df']]).copy()
                    if st.session_state.get('uploaded',False):
                        st.session_state['txn_df'].to_csv("data/txn_df.csv",index=False)
                    with open("data/client_list.txt", "w") as f:   
                        f.write("\n".join(st.session_state['client_list']))
                    st.session_state['uploaded'] = False
        if st.session_state.get("mod_txn_df") is not None and not st.session_state['has_final_file']:
            st.write("Stats for the final file:")
            st.dataframe(get_stats(st.session_state['mod_txn_df']).style.format(precision=0),use_container_width=True, hide_index=True,)
            # save to data
            
                
    if "is_exported" not in st.session_state:
        st.session_state['is_exported'] = False
        st.session_state['excel_buffer'] = b''
    if not st.session_state['has_final_file']:
        with button_cols[0]:
            if st.button("Export to Excel"):
                st.session_state['excel_buffer'] = export_txn_df_to_excel(
                    st.session_state['txn_df'],
                    pd.concat([st.session_state['mod_txn_df'], st.session_state['intercompany_txn']])[[
                        "Subsidiary: Name",
                        "Department: Name",
                        "Account (Line): Name",
                        "Accounting Period: Start Date",
                        "Document Number",
                        "Entity (Line)",
                        "ICC Projects",
                        "Amount",
                        "department_revised",
                        "client",
                        "project",
                        "Entity (Line): Internal ID",
                        "Entity (Line): HubSpot Deal ID",
                        "Match score",
                    ]],
                    st.session_state['client_list'],
                    numerical_cols=["Amount"],
                    null_fill_col="client",
                )
                st.session_state['is_exported'] = True

        with button_cols[1]:
            st.download_button(
                label="Download Excel",
                data=st.session_state['excel_buffer'],
                file_name="netsuite_revenue.xlsx",
                mime="application/vnd.ms-excel",
                disabled=not st.session_state['is_exported']
            )



with button_cols[0]:
    if st.session_state.get('uploaded_final_file'):
        st.button("Confirm Revenues", on_click=insert_revenue_report, args=(st.session_state['final_txn_df'],))
