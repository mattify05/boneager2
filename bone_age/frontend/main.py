#!/usr/bin/env python3

import streamlit as st
from app import home, analysis, login

if st.session_state.get("force_rerun"):
    del st.session_state["force_rerun"]
    st.rerun()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "show_login" not in st.session_state:
    st.session_state.show_login = False

if not st.session_state.authenticated:
    if st.session_state.show_login:
        login.login_form()
    else:
        login.register_user()
else:   
    st.button("Logout", on_click=login.logout)

    st.set_page_config(
        page_title="Bone-Ager", 
        page_icon=":bone:",
        layout="wide",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )

    home.main_ui()

    if st.session_state.get("uploaded_file"):
        analysis.display()
