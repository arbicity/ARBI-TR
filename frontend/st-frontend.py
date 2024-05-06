import streamlit as st
from tools import load_languages, download_youtube_video, process_file, poll_status
import os
import pandas as pd

# Streamlit UI setup
st.title('ARBI Transcribe and Translate')
languages_file_path = 'languages.txt'
languages = load_languages(languages_file_path)
speaker_options = ['*Autodetect', '1', '2', '3', '4', '5', '6', '7']

with st.sidebar:
    st.header('Settings')
    size_of_model = st.selectbox("Model Size", ["small", "large"], index=1)
    task_str = st.selectbox("Task", ["transcribe", "translate"], index=0)
    source_language = st.selectbox("Source Language", options=languages, index=languages.index('*Autodetect'))
    speaker_number = st.selectbox("Number of Speakers", options=speaker_options, index=0)

st.write("Step 1.")
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_files = st.file_uploader("Upload audio/video file(s)", type=['wav', 'mp3', 'mp4', 'm4a'], accept_multiple_files=True)

with col2:
    youtube_url = st.text_area("or enter YouTube Video URLs, one per line")
st.write("Step 2.")
downloaded_files = []
if st.button('Generate Transcript'):
    session_ids = []
    for file in uploaded_files + downloaded_files:
        result = process_file(file, file.name, size_of_model, task_str, source_language, '*Autodetect')
        session_id = result.get('session_id')  # Get the session_id from the result
        if session_id:
            session_ids.append(session_id)
        if 'error' in result:
            st.error(result['error'])
        else:
            st.success(result['message'])
            expander = st.expander(f"Click to view and download the transcript for {file.name}")
            with expander:
                for update in poll_status(session_id, file.name):
                    if isinstance(update, pd.DataFrame):
                        st.dataframe(update)
                    else:
                        st.write(update)

if youtube_url:
    urls = youtube_url.split('\n')
    for url in urls:
        if url.strip():
            with st.spinner(f'Downloading YouTube video from {url}...'):
                try:
                    downloaded_file = download_youtube_video(url)
                    downloaded_files.append(downloaded_file)
                    _, file_ext = os.path.splitext(downloaded_file)
                    if file_ext in ['.mp4', '.m4a']:
                        st.video(downloaded_file)
                    else:
                        st.audio(downloaded_file)
                except Exception as e:
                    st.error(f"Failed to download video from {url}. Error: {e}")

if uploaded_files:
    for file in uploaded_files:
        display_file = file.getvalue()
        _, file_ext = os.path.splitext(file.name)
        if file_ext in ['.mp4', '.m4a']:
            st.video(display_file)
        else:
            st.audio(display_file)

