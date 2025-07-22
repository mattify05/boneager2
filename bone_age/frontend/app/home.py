#!/usr/bin/env python3

import streamlit as st

def main_ui():
    st.title("Bone-Ager")
    st.subheader("An automatic paediatric bone age assessment tool")

    st.text_input("What is your name?", key="name")

    option = st.selectbox('What is the sex?', ('Female', 'Male', 'Unknown'))
    st.write('You selected:', option)

    st.sidebar.header('Home')
    st.sidebar.subheader('Our Github')
    st.sidebar.subheader('About Us')
    st.sidebar.subheader('Contact Us')
        
    uploaded = st.file_uploader(
        "Upload X-ray image (JPEG, PNG, or DICOM)", 
        type=["jpeg", "jpg", "png", "dcm"]
    )

    if uploaded:
        st.session_state.uploaded_file = uploaded
