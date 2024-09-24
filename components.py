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
    pages = st_navbar(
        pages=["Analysis", "Forecast Results", "TopDown"],
        # logo="🚀",
    )
    return pages
