import streamlit as st
from src.transcribe_ui import transcribe_tab

# Streamlit UI setup
st.set_page_config(
    page_title="ARBI Assistant",
    page_icon="robot_face",
    layout="centered")

# Define the tabs
tab1, tab2 = st.tabs(["Transcribe", "My Documents"])

with tab1:
    transcribe_tab()

with tab2:
    st.write("Other features or tabs can be added here.")
