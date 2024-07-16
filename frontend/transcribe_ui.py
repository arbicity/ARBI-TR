import os
import pandas as pd
import streamlit as st
from transcribe_tools import load_languages, download_youtube_video, process_file, poll_status

def transcribe_tab():
    """
    Constructs the UI components and handles file and YouTube video operations for transcription.
    """
    api_key = st.sidebar.text_input("Enter your API Key:", type="password")
    
    col1, col2 = st.columns([.2, .8])
    with col1:
        st.image("ARBI_Assistant.png", width=75)
    with col2:
        st.title('Transcribe and Translate')

    with st.expander(label="About this tool"):
        with open("about.md", "r", encoding="utf-8") as file:
            st.markdown(file.read())

    uploaded_files, downloaded_files = setup_file_and_video_download()

    settings = setup_transcription_settings()
    if st.button('Generate Transcript'):
        if api_key:
            process_files_for_transcription(uploaded_files, downloaded_files, settings, api_key)
        else:
            st.error("API Key is required to process the transcription.")

def setup_file_and_video_download():
    """
    Sets up columns for uploading files and entering YouTube URLs for downloading.
    Returns tuples of uploaded files and downloaded file information.
    """
    col1, col2 = st.columns([1, 1])
    uploaded_files = upload_files_ui(col1)
    downloaded_files = download_youtube_ui(col2)
    return uploaded_files, downloaded_files

def upload_files_ui(column):
    """
    Provides a UI for uploading files and previews them directly.
    """
    with column:
        uploaded_files = st.file_uploader(
            "Upload audio/video file(s)", type=['wav', 'mp3', 'mp4', 'm4a'],
            accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                display_media(file)
        return uploaded_files or []

def download_youtube_ui(column):
    """
    Provides a UI for downloading YouTube videos and displays them.
    Manages downloads across reruns using Streamlit's session state.
    """
    with column:
        youtube_url = st.text_area("or enter YouTube Video URLs, one per line")
        if 'downloaded_files' not in st.session_state:
            st.session_state['downloaded_files'] = []

        downloaded_files = []
        if youtube_url:
            urls = youtube_url.split('\n')
            for url in urls:
                if url.strip() and not any(url == d[2] for d in st.session_state['downloaded_files']):
                    try:
                        video_file_path, video_file_name = download_youtube_video(url)
                        downloaded_files.append((video_file_path, video_file_name, url))
                        display_media(open(video_file_path, 'rb'), os.path.splitext(video_file_path)[1])
                    except Exception as e:
                        st.error(f"Failed to download video from {url}. Error: {e}")
            # Update session state with new downloads
            st.session_state['downloaded_files'].extend(downloaded_files)
        return [file for file in st.session_state['downloaded_files'] if file[2] in youtube_url.split('\n')]

def display_media(media, file_ext=None):
    """
    Displays video or audio based on file extension or inferred from media object.
    """
    if not file_ext:
        file_ext = os.path.splitext(media.name)[1]
    if file_ext in ['.mp4', '.m4a']:
        st.video(media.getvalue() if hasattr(media, 'getvalue') else media)
    else:
        st.audio(media.getvalue() if hasattr(media, 'getvalue') else media)

def setup_transcription_settings():
    """
    Allows users to configure settings for transcription within an expander.
    """
    with st.expander("Settings"):
        languages_file_path = 'languages.txt'
        languages = load_languages(languages_file_path)
        speaker_options = ['*Autodetect', '1', '2', '3', '4', '5', '6', '7', '8']
        settings = {
            'size_of_model': st.selectbox(
                "Speech recognition model size (small is faster, large is more accurate)",
                ["small", "large"], index=1),
            'task_str': st.selectbox("Task", ["transcribe", "translate"], index=0),
            'source_language': st.selectbox(
                "Source Language", options=languages,
                index=languages.index('english') if 'english' in languages else 0),
            'speaker_number': st.selectbox("Number of Speakers", options=speaker_options, index=0)
        }
        return settings

def process_files_for_transcription(uploaded_files, downloaded_files, settings, api_key):
    """
    Processes each file for transcription based on the provided settings and displays results.
    """
    session_ids = []
    for files in (uploaded_files, downloaded_files):
        for file in files:
            try:
                if hasattr(file, 'getvalue'):  # Uploaded files
                    file_content = file.getvalue()
                    file_name = file.name
                else:  # Downloaded files, which are tuples
                    with open(file[0], 'rb') as f:
                        file_content = f.read()
                    file_name = file[1]

                result = process_file(
                    file_content, file_name, settings['size_of_model'],
                    settings['task_str'], settings['source_language'],
                    settings['speaker_number'], api_key)
                session_id = result.get('session_id')
                if session_id:
                    session_ids.append(session_id)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success(result['message'])
                    display_transcript_results(session_id, api_key)
            except Exception as e:
                st.error(f"Error processing file {file_name}: {e}")
    return session_ids

def display_transcript_results(session_id, api_key):
    """
    Displays transcript results for a session in an expander.
    """
    expander = st.expander(f"Session {session_id}: Transcript Ready!")
    with expander:
        for update in poll_status(session_id, f"Session {session_id}", api_key):
            if isinstance(update, pd.DataFrame):
                st.dataframe(update)
            else:
                st.write(update)
