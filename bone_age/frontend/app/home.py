#!/usr/bin/env python3

import streamlit as st
from app import helpers

def main_ui():
    st.title("Bone-Ager")
    st.subheader("An automatic paediatric bone age assessment tool")

    st.markdown(
        """
        #### :bone: How to Use Bone-Ager

        1. **Upload** one or more X-ray images (JPEG, PNG, or DICOM).
        2. **Enter patient details** for each image
        3. **Submit** the details to start the automatic analysis.
        4. **Review** your results on-screen.
        5. **Download** a detailed report whenever you want.

        > Note: *If you prefer not to provide patient info, just skip it — the analysis will still run smoothly based on the images.*
        """
    )

    uploaded = st.file_uploader(
        "Upload X-ray image (JPEG, JPG, PNG, or DICOM)", 
        type=["jpeg", "jpg", "png", "dcm"],
        accept_multiple_files=True,
        on_change=helpers.reset_analysis
    )

    if uploaded:
        st.session_state.uploaded_file = uploaded
        st.session_state.analysis_done = False

        st.info(
            "Files uploaded successfully! Please fill out the patient metadata form below "
            "and then submit to start the bone age analysis"
        )
    else:
        if "uploaded_file" in st.session_state:
            del st.session_state["uploaded_file"]
            st.session_state.analysis_done = False

