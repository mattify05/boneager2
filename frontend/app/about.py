#!/usr/bin/env python3

import streamlit as st

def render_about():
    st.title("Bone-Ager")

    st.write("""
        Our project primarily targets clinicians and healthcare providers, 
        aiming to develop a locally runnable pediatric bone age classifier that has 
        been trained using a collection of paediatric hand X-rays from the Radiological
        Society of North America (RSNA).
    """)
    
    st.subheader("""
        An accessible and transparent solution for paediatric bone age assessment
    """)
       
    st.write("""
        Assessing a child’s bone age is critical for diagnosing growth disorders and 
        guiding treatment decisions. Yet existing automated tools remain expensive, 
        opaque, or limited in scope. 
    """)

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("images/hand_xrays.png", width=700)

    st.subheader("Model Training")

    st.write("""
        Our model was trained using a multitasking learning approach, where the convolutional
        neural netwok (CNN) processes X-rays through multiple epochs. An epoch represents
        one complete pass through the entire training dataset. During training, the CNN
        backbone extracts features from preprocessed images which are fed into three output
        heads, each responsible for bone age regression, gender classification and
        estimating the uncertainty levels.
    """)

    st.write("""
        The figure below illustrates the model's performance over 50 epochs
        using loss and mean absolute error (MAE). These results highlight the effectiveness
        of thorough training, clean data and hyperparameter tuning, supporting our choice
        of regression for bone age estimation. Future work will focus on subgroup analysis
        and further improvements through data augmentation and ensemble methods, marking 
        a key milestone for Bone-Ager's clinical applicability
    """)

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("images/Training_MAE_figs.png", caption="Training timeline with MAE", width=700)