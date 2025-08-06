#!/usr/bin/env python3

import streamlit as st
from app import helpers

def main_ui():
    st.title("Bone-Ager")
    st.subheader("An automatic paediatric bone age assessment tool")

    st.text_input("What is the patient's name?", key="patient_name")

    st.text_input("What is the patient ID?", key="patient_id")

    option = st.selectbox('What is the sex?', ('Female', 'Male', 'Unknown'))
    st.write('You selected:', option)
        
    uploaded = st.file_uploader(
        "Upload X-ray image (JPEG, JPG, PNG, or DICOM)", 
        type=["jpeg", "jpg", "png", "dcm"],
        accept_multiple_files=True,
        on_change=helpers.reset_analysis
    )

    if uploaded:
        st.session_state.uploaded_file = uploaded
    else:
        if "uploaded_file" in st.session_state:
            del st.session_state["uploaded_file"]
