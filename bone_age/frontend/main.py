
#!/usr/bin/env python3

import streamlit as st
import pydicom
from PIL import Image
import numpy as np
import pandas as pd
import time

st.title("Bone-Ager")
st.subheader("An automatic paediatric bone age assessment tool")

st.text_input("What is your name?", key="name")

option = st.selectbox('What is the gender?', ('Female', 'Male', 'Unknown'))
st.write('You selected:', option)

st.sidebar.header('Home')
st.sidebar.subheader('Our Github')
st.sidebar.subheader('About Us')
st.sidebar.subheader('Contact Us')
    
uploaded_file = st.file_uploader(
    "Upload X-ray image (JPEG, PNG, or DICOM)", 
    type=["jpeg", "jpg", "png", "dcm"]
)

# Utility: normalize pixel values to 0-255 for display
def normalize_to_uint8(image):
    if np.max(image) == np.min(image):
        return np.zeros_like(image, dtype=np.uint8)
    image = image.astype(np.float32)
    image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    return image.astype(np.uint8)

# Placeholder function for bone age prediction
def estimate_bone_age(image_array):
    # TODO: Replace this with your ML model
    return 12.5  

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    file_ext = uploaded_file.name.lower().split('.')[-1]

    st.write('Starting analysis...')
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.1)
    
    st.write('...and now we\'re done!'
             )
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


