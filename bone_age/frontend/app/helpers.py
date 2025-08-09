#!/usr/bin/env python3

import streamlit as st
import numpy as np
import tempfile
import cv2
from PIL import Image
import io
import csv
import threading
import time

import sys
from pathlib import Path

# Add bone_age/ to the sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from predictor import FlexibleBoneAgePredictor

# Utility: normalize pixel values to 0-255 for display
def normalize_to_uint8(image):
    if np.max(image) == np.min(image):
        return np.zeros_like(image, dtype=np.uint8)
    image = image.astype(np.float32)
    image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    return image.astype(np.uint8)

# Accesses the best_bone_age_model.pth file
@st.cache_resource
def get_predictor():
    model_path = Path(__file__).resolve().parent.parent.parent / "best_bone_age_model.pth"
    predictor = FlexibleBoneAgePredictor(str(model_path))
    return predictor

# Uses the get_predictor helper function to load the model and return the pretrained
# bone age prediction results
def estimate_bone_age(image_array, gender=None, use_tta=True):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, image_array)

    predictor = get_predictor()
    result = predictor.predict_single_image(temp_path, gender=gender, use_tta=use_tta)

    if isinstance(result, list):
        result = result[0]
    
    return {
        "predicted_age_months": round(result.predicted_age_months, 1),
        "predicted_age_years": round(result.predicted_age_years, 1),
        "confidence": round(result.confidence_score, 2),
        "uncertainty_months": round(result.uncertainty, 1)
    }

# Maps the sex (female, male, unknown) to an integer (0, 1, None) respectively for
# the model to interpret
def map_sex_format(sex_str):
    sex_str = sex_str.lower()
    if sex_str == 'female':
        return 0
    elif sex_str == 'male':
        return 1
    else:
        return None

# Converts uploaded image bytes to a numpy array 
def decode_image(uploaded_file_bytes):
    if not uploaded_file_bytes:
        raise ValueError("Uploaded file is empty.")
    try:
        image = Image.open(io.BytesIO(uploaded_file_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except:
        raise ValueError("Uploaded file is not a valid image.")

# Converts the dataframe to a csv file, so it can be downloaded
@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")

# clears the analysis_done session state 
def reset_analysis():
    if "analysis_done" in st.session_state:
        del st.session_state["analysis_done"]

# Writes the bone age results to a CSV file (bone_age_results.csv)
def save_results_to_csv(results, filename="bone_age_results.csv"):
    keys = results[0].keys()

    with open(filename, mode="w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

# appends the results of each image to the results list and returns it 
def batch_process_images(image_arrays):
    results = []
    for image in image_arrays:
        result = estimate_bone_age(image)
        results.append(result)
    return results

# Uses threads to allow for concurrent execution of code to speed up analysis
def progress_using_threads(image_array, estimate_fn, progress_callback=None):
    result_container = {"result": None}

    def run_analysis(image_array, container):
        container["result"] = estimate_bone_age(image_array)
    
    thread = threading.Thread(target=run_analysis, args=(image_array, result_container))
    thread.start()

    progress = 0
    while thread.is_alive():
        time.sleep(0.1)
        progress = min(progress + 2, 100)
        if progress_callback:
            progress_callback(progress)
    if progress_callback:
        progress_callback(100)

    return result_container["result"]