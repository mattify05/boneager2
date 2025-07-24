#!/usr/bin/env python3

import streamlit as st

def render_about():
    st.title("About Us")
    st.write("""
    Bone-Ager is an automatic bone age assessment tool that has been trained 
    using the RSNA dataset""")