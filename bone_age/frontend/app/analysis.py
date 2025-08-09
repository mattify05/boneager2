#!/usr/bin/env python3

import streamlit as st
import pydicom
from PIL import Image
import numpy as np
import pandas as pd

from app import helpers

def display():
    uploaded_file = st.session_state.get("uploaded_file")
    if not uploaded_file:
        return

    if not isinstance(uploaded_file, list):
        uploaded_files = [uploaded_file]
    else:
        uploaded_files = uploaded_file

    if "metadata_submitted" not in st.session_state:
        st.session_state.metadata_submitted = False

    # Show metadata inputs always, prefilled from session_state or empty string
    with st.form("metadata_form"):
        for i, uploaded_file in enumerate(uploaded_files):
            st.write(f"**Metadata for file {i + 1}**: {uploaded_file.name}")
            # Use get with default empty string so inputs are persistent
            name_val = st.session_state.get(f"name_{i}", "")
            id_val = st.session_state.get(f"id_{i}", "")
            sex_val = st.session_state.get(f"sex_{i}", "Unknown")

            # Show inputs with values
            name = st.text_input(f"Patient Name #{i + 1}", value=name_val, key=f"name_{i}")
            pid = st.text_input(f"Patient ID #{i + 1}", value=id_val, key=f"id_{i}")
            sex = st.selectbox(f"Sex #{i + 1}", options=["Female", "Male", "Unknown"], index=["Female", "Male", "Unknown"].index(sex_val), key=f"sex_{i}")

        submitted = st.form_submit_button("Submit")

    if submitted:
        st.session_state.metadata_submitted = True
        st.session_state.analysis_done = False

    # Run analysis only if submitted and not already done
    if st.session_state.metadata_submitted and not st.session_state.get("analysis_done", False):
        results = []
        for index, uploaded_file in enumerate(uploaded_files):
            file_ext = uploaded_file.name.lower().split('.')[-1]

            bar = st.progress(0, text="Starting analysis...")

            col1, col2, col3 = st.columns([2, 3, 2])

            if file_ext == "dcm":
                dicom_data = pydicom.dcmread(uploaded_file)
                for tag in ["PatientName", "PatientID", "PatientBirthDate"]:
                    if tag in dicom_data:
                        dicom_data.data_element(tag).value = ""

                image = dicom_data.pixel_array
                image = helpers.normalize_to_uint8(image)

                with col1:
                    st.image(image, caption="DICOM Image Preview", use_container_width=True)

                uploaded_file.seek(0)

            else:
                with col1:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Image Preview", use_container_width=True)
                    image = np.array(image)

            sex_mapped = helpers.map_sex_format(sex)

            result = helpers.progress_using_threads(
                image_array=image,
                estimate_fn=lambda img: helpers.estimate_bone_age(img, gender=sex_mapped, use_tta=True),
                progress_callback=lambda p: bar.progress(p, text="Analysing...")
            )

            st.success("Analysis complete.")

            # Metadata from session_state (inputs are stored automatically)
            patient_name = st.session_state.get(f"name_{index}", f"Patient {index + 1}")
            patient_id = st.session_state.get(f"id_{index}", f"ID_{index + 1}")
            sex = st.session_state.get(f"sex_{index}", "Unknown")

            result["patient_name"] = patient_name
            result["patient_id"] = patient_id
            result["sex"] = sex
            results.append(result)

            with col2:
                st.write(f"Patient Name: **{patient_name}**")
                st.write(f"Patient ID: **{patient_id}**")
                st.write(f"Sex: **{sex}**")

            with col3:
                st.success(f"Estimated Bone Age: **{result['predicted_age_months']} months ({result['predicted_age_years']} years)**")
                st.success(f"Confidence: **{result['confidence'] * 100:.2f}%**")
                st.success(f"Uncertainty: **± {result['uncertainty_months']} months**")

        if results:
            df = pd.DataFrame(results)
            csv = helpers.convert_for_download(df)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="bone_age_results.csv",
                mime="text/csv",
                icon=":material/download:",
            )
        st.session_state.analysis_done = True

      # Show "Analyze another image" button after analysis is done
    if st.session_state.get("analysis_done", False):
        st.write("")
        st.write("Ready to analyse another X-ray? :point_down:")
        if st.button("Start New Analysis"):
            # Clear session state related to files and analysis
            for key in ["uploaded_file", "metadata_submitted", "analysis_done"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Increment upload_count to reset uploader widget
            st.session_state.upload_count += 1
            st.rerun()