#!/usr/bin/env python3

import streamlit as st

st.header("Settings")
st.write(f"You are logged in as {st.session_state.role}.")
