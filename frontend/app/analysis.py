#!/usr/bin/env python3

import streamlit as st
import pydicom
from PIL import Image
import numpy as np
import pandas as pd

from app import helpers
from dataclasses import asdict
from bone_age.predictor import PredictionResult
import time
from pathlib import Path
from bone_age.predictor import get_predictor
import os

# Initialize predictor only once using session state:cite[3]:cite[8]
if 'bone_age_predictor' not in st.session_state:
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "models" / "best_bone_age_model.pth"
    
    print(f"🔍 Initializing predictor with path: {model_path}")
    print(f"🔍 Path exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()
    
    # Initialize and store in session state
    st.session_state.bone_age_predictor = get_predictor(
        model_path=str(model_path), 
        device="auto"
    )
    print(f"✅ Predictor initialized and stored in session state")

# Get the predictor from session state
predictor = st.session_state.bone_age_predictor

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
        # Clear previous results when resubmitting
        if "analysis_results" in st.session_state:
            del st.session_state.analysis_results

    # only runs analysis if submitted and not already done and no results exist
    if (st.session_state.metadata_submitted and 
        not st.session_state.get("analysis_done", False) and 
        "analysis_results" not in st.session_state):
        
        results = []
        for index, uploaded_file in enumerate(uploaded_files):

            # --- Get metadata from session (define BEFORE using later) ---
            patient_name = st.session_state.get(f"name_{index}", f"Patient {index + 1}")
            patient_id   = st.session_state.get(f"id_{index}", f"ID_{index + 1}")
            sex_label    = st.session_state.get(f"sex_{index}", "Unknown")

            file_ext = uploaded_file.name.lower().split('.')[-1]

            bar = st.progress(0, text="Starting analysis...")

            # --- Load image (same as you had) ---
            if file_ext == "dcm":
                dicom_data = pydicom.dcmread(uploaded_file)
                for tag in ["PatientName", "PatientID", "PatientBirthDate"]:
                    if tag in dicom_data:
                        dicom_data.data_element(tag).value = ""
                image = dicom_data.pixel_array
                image = helpers.normalize_to_uint8(image)
                uploaded_file.seek(0)
            else:
                image = Image.open(uploaded_file)
                image = np.array(image)

            # --- Map sex to predictor input; force a single-output mode for UI ---
            sex_for_pred = helpers.map_sex_format(sex_label)
            if sex_for_pred in (None, "Unknown", "unknown"):
                sex_for_pred = "average"  # ensures a single PredictionResult, not a list

            # --- Small progress animation (keeps UI responsive) ---
            for p in range(0, 101, 10):
                time.sleep(0.01)
                bar.progress(p, text="Analysing...")

            # --- Call the estimator directly (no threads) ---
            try:
                result = helpers.estimate_bone_age(image, gender=sex_for_pred, use_tta=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

            st.success("Analysis complete.")

            # --- Normalize result to a dict your UI expects ---
            # If predictor returns both genders somehow, pick the first
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            # Convert dataclass -> dict
            if isinstance(result, PredictionResult):
                result = asdict(result)

            # Ensure we have a dict
            if not isinstance(result, dict):
                st.error(f"Unexpected result type: {type(result)}")
                return

            # Harmonize key names used later in the UI
            if "confidence" not in result and "confidence_score" in result:
                result["confidence"] = result["confidence_score"]
            if "uncertainty_months" not in result and "uncertainty" in result:
                result["uncertainty_months"] = result["uncertainty"]

            # Final guard
            if result is None:
                st.error("Prediction failed: no result. Check the model path and image")
                return

            # --- Attach metadata and store ---
            result["patient_name"] = patient_name
            result["patient_id"]   = patient_id
            result["sex"]          = sex_label
            results.append(result)


        st.session_state.analysis_results = results
        st.session_state.analysis_done = True
        st.rerun()  # Force a rerun to display results

    if "analysis_results" in st.session_state:
        results = st.session_state.analysis_results
        
        for index, result in enumerate(results):
            col1, col2, col3 = st.columns([2, 3, 2])
            
            uploaded_file = uploaded_files[index]
            file_ext = uploaded_file.name.lower().split('.')[-1]
            
            with col1:
                if file_ext == "dcm":
                    dicom_data = pydicom.dcmread(uploaded_file)
                    image = dicom_data.pixel_array
                    image = helpers.normalize_to_uint8(image)
                    st.image(image, caption="DICOM Image Preview", use_container_width=True)
                else:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Image Preview", use_container_width=True)

            with col2:
                st.write(f"Patient Name: **{result['patient_name']}**")
                st.write(f"Patient ID: **{result['patient_id']}**")
                st.write(f"Sex: **{result['sex']}**")

            with col3:
                st.success(f"Estimated Bone Age: **{result['predicted_age_months']} months ({result['predicted_age_years']} years)**")
                st.success(f"Confidence: **{result['confidence'] * 100:.2f}%**")
                st.success(f"Uncertainty: **± {result['uncertainty_months']} months**")

        # Download button
        df = pd.DataFrame(results)
        csv = helpers.convert_for_download(df)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="bone_age_results.csv",
            mime="text/csv",
            icon=":material/download:",
        )

        st.write("")
        st.write("Ready to analyse another X-ray? :point_down:")
        if st.button("Start New Analysis"):
            # Clear session state related to files and analysis
            for key in ["uploaded_file", "metadata_submitted", "analysis_done", "analysis_results"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Increment upload_count to reset uploader widget
            st.session_state.upload_count += 1
            st.rerun()