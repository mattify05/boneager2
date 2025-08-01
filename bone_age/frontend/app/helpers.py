#!/usr/bin/env python3

import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import cv2
from PIL import Image
import io

import sys
from pathlib import Path

# Add bone_age/ to the sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from simple_predict import load_model, predict_bone_age

# Utility: normalize pixel values to 0-255 for display
def normalize_to_uint8(image):
    if np.max(image) == np.min(image):
        return np.zeros_like(image, dtype=np.uint8)
    image = image.astype(np.float32)
    image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    return image.astype(np.uint8)

@st.cache_resource
def get_model():
    model_path = Path(__file__).resolve().parent.parent.parent.parent / "checkpoint_epoch_51.pth" 
    model, device = load_model(str(model_path))
    return model, device

def estimate_bone_age(image_array):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, image_array)
    
    model, device = get_model()

    result = predict_bone_age(temp_path, model, device, monte_carlo_samples=5)
    return {
        'predicted_age_months': round(result['age_months'], 1),
        'predicted_age_years': round(result['age_years'], 1),
        'confidence': round(result['confidence'], 2),
        'uncertainty_months': round(result['uncertainty'], 1),
        'development_stage': result['stage']
    }  

def decode_image(uploaded_file_bytes):
    """Converts uploaded image bytes into a NumPy array (OpenCV format)."""
    image = Image.open(io.BytesIO(uploaded_file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def get_data():
    df = pd.DataFrame(
        np.random.randn(50, 20), columns=("col %d" % i for i in range(20))
    )
    return df

# Converts the dataframe to a csv file, so it can be downloaded
@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")

# clears the analysis_done session state 
def reset_analysis():
    if "analysis_done" in st.session_state:
        del st.session_state["analysis_done"]