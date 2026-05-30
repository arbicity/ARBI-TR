import streamlit as st
from transcribe_ui import transcribe_tab


def main():
    # Streamlit UI setup
    st.set_page_config(
        page_title="ARBI Assistant",
        page_icon="🤖",  # Ensure the icon renders or use a URL for an image
        layout="centered",
    )

    # Define the tabs
    tab1, tab2 = st.tabs(["Transcribe", "My Documents"])

    with tab1:
        transcribe_tab()  # Ensures the Transcribe functionality is encapsulated within this tab

    with tab2:
        st.write("Other features or tabs can be added here.")


# Ensure the app runs when directly invoked
if __name__ == "__main__":
    main()
