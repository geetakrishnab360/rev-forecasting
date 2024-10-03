import os
import base64
import streamlit as st
from streamlit_navigation_bar import st_navbar


def set_header(heading):
    path = os.path.dirname(__file__)
    file_ = open(f"{path}/images.png", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    return st.markdown(
        f"""<div class='main-header'>
                    <h1>{heading}</h1>
                    <img class='logo' src="data:image;base64,{data_url}", alt="Logo">
            </div>""",
        unsafe_allow_html=True,
    )


def set_navigation():
    st.write("")
    button_container = st.container()
    with button_container:
        buttons = st.columns([1, 1, 1])
        if buttons[0].button('Pipeline Analysis', use_container_width=True):
            st.switch_page(r"./Analysis.py")
        if buttons[1].button('Pipeline Forecast', use_container_width=True):
            st.switch_page(r"./pages/1_Forecast_Results.py")
        if buttons[2].button('TopDown Forecast', use_container_width=True):
            st.switch_page(r"./pages/2_TopDown.py")
    hide_sidebar_style = """
        <style>
        [data-testid="stSidebarCollapsedControl"] {display: none;}
        [data-testid="stSidebar"] {display: none;}
        </style>
    """
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)
