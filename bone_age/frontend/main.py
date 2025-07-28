#!/usr/bin/env python3

import streamlit as st
from app import home, analysis, login, about, contact

if st.session_state.get("force_rerun"):
    del st.session_state["force_rerun"]
    st.rerun()

st.set_page_config(
        page_title="Bone-Ager", 
        page_icon=":bone:",
        layout="wide",
        # can update the menu items 
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "show_login" not in st.session_state:
    st.session_state.show_login = False

# sets the default page to be the homepage
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

current_page = st.session_state.current_page

if not st.session_state.authenticated:
    if st.session_state.show_login:
        login.login_form()
    else:
        login.register_user()
else:
    st.write(f"Welcome {st.session_state.get('user', 'User')}!")   
    st.button("Logout", on_click=login.logout)

    if current_page == "Home":
        home.main_ui()
        if st.session_state.get("uploaded_file") and not st.session_state.get("analysis_done"):
            analysis.display()
            st.session_state.analysis_done = True
    elif current_page == "About":
        about.render_about()
    elif current_page == "Contact":
        contact.render_contact()

    st.sidebar.title("Navigation")

    if st.sidebar.button("Home", use_container_width=True):
        st.session_state.current_page = "Home"
        st.rerun()

    if st.sidebar.button("About Us", use_container_width=True):
        st.session_state.current_page = "About"
        st.rerun()

    if st.sidebar.button("Contact Us", use_container_width=True):
        st.session_state.current_page = "Contact"
        st.rerun()
    
    st.sidebar.markdown("---")   
    st.sidebar.markdown("[Visit our GitHub :link:]"
    "(https://github.com/jjjaden-hash/DESN2000-BINF-M13B_GAMMA/tree/main)")
    st.sidebar.markdown("---")

