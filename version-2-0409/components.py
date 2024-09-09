import os
import base64
import streamlit as st


def set_header(heading):
    path = os.path.dirname(__file__)
    file_ = open(f"{path}/logo.png", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    return st.markdown(
        f"""<div class='main-header'>
                    <h1>{heading}</h1>
                    <img class='logo' src="data:image;base64,{data_url}", alt="Logo">
            </div>""",
        unsafe_allow_html=True,
    )
