#!/usr/bin/env python3

import streamlit as st
import streamlit_permalink as stp
import pydicom
from PIL import Image
import numpy as np
import pandas as pd
import time

from app.helpers import normalize_to_uint8, estimate_bone_age, get_data, convert_for_download

def display():
    uploaded_file = st.session_state.get("uploaded_file")
    if not uploaded_file:
        return

    st.success("File uploaded successfully!")
    file_ext = uploaded_file.name.lower().split('.')[-1]

    st.write('Starting analysis...')
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
        image = normalize_to_uint8(image)

        st.image(image, caption="DICOM Image Preview", use_container_width=True)

    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image Preview", use_container_width=True)
        image = np.array(image)  # convert for processing

    # Bone age estimation (placeholder)
    bone_age = estimate_bone_age(image)
    st.success(f"Estimated Bone Age: {bone_age} years")

    # converts the data to csv for download and implements the download button 
    df = get_data()
    csv = convert_for_download(df)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="data.csv",
        mime="text/csv",
        icon=":material/download:",
    )


