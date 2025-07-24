#!/usr/bin/env python3

import streamlit as st
import numpy as np
import pandas as pd


# Utility: normalize pixel values to 0-255 for display
def normalize_to_uint8(image):
    if np.max(image) == np.min(image):
        return np.zeros_like(image, dtype=np.uint8)
    image = image.astype(np.float32)
    image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    return image.astype(np.uint8)

# Placeholder function for bone age prediction
def estimate_bone_age(image_array):
    # TODO: Replace with our ML model
    return 12.5  

def get_data():
    df = pd.DataFrame(
        np.random.randn(50, 20), columns=("col %d" % i for i in range(20))
    )
    return df

# Converts the dataframe to a csv file, so it can be downloaded
@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")
