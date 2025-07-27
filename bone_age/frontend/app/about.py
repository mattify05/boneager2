#!/usr/bin/env python3

import streamlit as st

def render_about():
    st.title("Bone-Ager")

    st.write("""Our project primarily targets clinicians and healthcare providers, 
    aiming to develop a locally runnable pediatric bone age classifier that has 
    been trained using the RSNA Bone Age dataset.""")
    
    st.subheader("""
    An accessible and transparent solution for paediatric bone age assessment""")
       
    st.write("""
    Assessing a child’s bone age is critical for diagnosing growth disorders and 
    guiding treatment decisions. Yet existing automated tools remain expensive, 
    opaque, or limited in scope. """)

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("images/hand_xrays.png")

    st.subheader("Model Training")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("images/Training_MAE_figs.png", caption="Training timeline with MAE")