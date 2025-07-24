#!/usr/bin/env python3

import streamlit as st
import hashlib
import yaml
import os

USER_DB = "users.yaml"

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

# Reads the user database from the YAML file and returns a dictionary of users
def load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r") as f:
            return yaml.safe_load(f).get("users", {})
    return {}

# Uses sha256 to hash passwords for security
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Adds each new user and their hashed password to the database in users.yaml
def save_user(username, password):
    users = load_users()
    hashed = hash_password(password)
    users[username] = {"password": hashed}
    with open(USER_DB, "w") as f:
        yaml.dump({"users": users}, f)

# ------------------------------------------------------------------------------
# Registration Functions
# ------------------------------------------------------------------------------

# Displays the user registration form and handles account creation 
def register_user():
    st.subheader("Create an account")
    new_user = st.text_input("New Username", key="reg_user")
    new_pass = st.text_input("New Password", type="password", key="reg_pass")

    if st.button("Register"):
        users = load_users()
        if new_user in users:
            st.error("Username already exists")
        elif not new_user or not new_pass:
            st.warning("Username and password cannot be empty.")
        else:
            save_user(new_user, new_pass)
            st.success("User registered! Please log in.")
            st.session_state.show_login = True
    
    st.markdown("Already have an account? :point_down:")
    if st.button("Log in here"):
        st.session_state.show_login = True

# ------------------------------------------------------------------------------
# Log in/out Functions
# ------------------------------------------------------------------------------

# Renders the login form UI and offers a registration option if the user 
# does not have an account
def login_form():
    st.subheader(":closed_lock_with_key: Log In")
    st.text_input("Username", key="user")
    st.text_input("Password", type="password", key="passwd")

    if st.button("Login"):
        creds_entered()
    
    st.markdown("Don't have an account?")
    if st.button("Create one here"):
        st.session_state.show_login = False

# Logs the current user out by clearing the session state and any query parameters
def logout():
    for key in ["authenticated", "user", "passwd", "username", "show_login"]:
        if key in st.session_state:
            del st.session_state[key]
    st.query_params.clear()
    st.session_state["force_rerun"] = True

# ------------------------------------------------------------------------------
# Authentication Functions
# ------------------------------------------------------------------------------

# Verifies that the user has entered the correct username and password 
def creds_entered():
    users = load_users()
    user = st.session_state.get("user", "").strip()
    passwd = st.session_state.get("passwd", "").strip()
    hashed_pass = hash_password(passwd)

    if user in users and users[user]["password"] == hashed_pass:
        st.session_state["authenticated"] = True
        st.session_state["username"] = user

        st.query_params.update({"page": "Home"})
        st.rerun()

    else:
        st.session_state["authenticated"] = False
        if not user:
            st.warning("Please enter a username")
        if not passwd:
            st.warning("Please enter a password")
        else:
            st.error("Invalid Username/Password")

def authenticate_user():
    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        st.subheader("Login")
        st.text_input(
            label="Username:", 
            value="", 
            key="user", 
            on_change=creds_entered
        )
        st.text_input(
            label="Password:", 
            value="", 
            key="passwd", 
            type="password", 
            on_change=creds_entered
        )
        return False
    else:
        return True
