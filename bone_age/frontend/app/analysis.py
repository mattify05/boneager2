#!/usr/bin/env python3

import streamlit as st
import streamlit_permalink as stp
import pydicom
from PIL import Image
import numpy as np
import pandas as pd
import time

from app import helpers

def display():
    uploaded_file = st.session_state.get("uploaded_file")
    if not uploaded_file:
        return

    for index, uploaded_file in enumerate(uploaded_file):
        st.success(f"file {index + 1}: {uploaded_file.name} uploaded successfully!")
        file_ext = uploaded_file.name.lower().split('.')[-1]

        st.write(f"Starting analysis for file {index + 1}...")
        latest_iteration = st.empty()
        bar = st.progress(0)

        for i in range(100):
            bar.progress(i + 1)
            time.sleep(0.1)
        
        st.write('...and now we\'re done!')

        if file_ext == "dcm":
            # Load and anonymize DICOM
            dicom_data = pydicom.dcmread(uploaded_file)
            for tag in ["PatientName", "PatientID", "PatientBirthDate"]:
                if tag in dicom_data:
                    dicom_data.data_element(tag).value = ""

            # Get image data and normalize
            image = dicom_data.pixel_array
            image = helpers.normalize_to_uint8(image)

            st.image(image, caption="DICOM Image Preview", use_container_width=True)

        else:
            col1, col2, col3 = st.columns([2, 3, 2])
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Image Preview", use_container_width=True)
                image = np.array(image)  # convert for processing
        
        with col2:
            patient_name = st.session_state.get("patient_name", "Unknown")
            st.write(f"Patient Name: **{patient_name}**")

            patient_id = st.session_state.get("patient_id", "N/A")
            st.write(f"Patient ID: **{patient_id}**")

        with col3:
            uploaded_file.seek(0)  # reset pointer before reading again
            image = helpers.decode_image(uploaded_file.read())
            result = helpers.estimate_bone_age(image)

            st.success(f"Estimated Bone Age: **{result['predicted_age_months']} months ({result['predicted_age_years']} years)**")
            st.success(f"Confidence: **{result['confidence']}**")
            st.success(f"Uncertainty: **{result['uncertainty_months']} months**")
            st.success(f"Development Stage: **{result['development_stage']}**")

    # converts the data to csv for download and implements the download button 
    df = helpers.get_data()
    csv = helpers.convert_for_download(df)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="data.csv",
        mime="text/csv",
        icon=":material/download:",
    )


