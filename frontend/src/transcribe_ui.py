"""
Module for managing transcription services in a Streamlit application. This module handles
uploading of audio/video files, downloading media from YouTube, configuring transcription
settings, and processing the transcription and translation of media files.
"""

import os
import pandas as pd
import streamlit as st
from src.transcribe_tools import load_languages, download_youtube_video, process_file, poll_status


def transcribe_tab():
    """
    Constructs the UI components and handles file and YouTube video operations for transcription.
    """
    st.title('Transcribe and Translate')
    with st.expander(label="About this tool"):
        with open("about.md", "r", encoding="utf-8") as file:
            st.markdown(file.read())

    uploaded_files, downloaded_files = setup_file_and_video_download()

    settings = setup_transcription_settings()
    if st.button('Generate Transcript'):
        process_files_for_transcription(
            uploaded_files, downloaded_files, settings)


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
    """
    with column:
        youtube_url = st.text_area("or enter YouTube Video URLs, one per line")
        downloaded_files = []
        if youtube_url:
            urls = youtube_url.split('\n')
            for url in urls:
                if url.strip() and not any(
                    url == d[2] for d in st.session_state.get(
                        'downloaded_files', [])):
                    try:
                        video_file_path, video_file_name = download_youtube_video(
                            url)
                        with open(video_file_path, 'rb') as file:
                            file_ext = os.path.splitext(video_file_path)[1]
                            display_media(file, file_ext)
                        downloaded_files.append(
                            (video_file_path, video_file_name, url))
                    except Exception as e:
                        st.error(
                            f"Failed to download video from {url}. Error: {e}")
            st.session_state['downloaded_files'] = downloaded_files
        return downloaded_files


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
        speaker_options = ['*Autodetect', '1',
                           '2', '3', '4', '5', '6', '7', '8']
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


def process_files_for_transcription(
        uploaded_files,
        downloaded_files,
        settings):
    """
    Processes each file for transcription based on the provided settings.
    """
    session_ids = []
    for files in (uploaded_files, downloaded_files):
        for file in files:
            try:
                if hasattr(file, 'getvalue'):  # Uploaded files
                    file_content = file.getvalue()
                else:  # Downloaded files
                    with open(file[0], 'rb') as f:
                        file_content = f.read()
                result = process_file(
                    file_content, file.name, settings['size_of_model'],
                    settings['task_str'], settings['source_language'],
                    settings['speaker_number'])
                session_id = result.get('session_id')
                if session_id:
                    session_ids.append(session_id)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success(result['message'])
                    display_transcript_results(session_id)
            except Exception as e:  # General exception due to multiple possible errors
                st.error(
                    f"Error processing file {file[1] if isinstance(file, tuple) else file.name}: {e}")
    return session_ids


def display_transcript_results(session_id):
    """
    Displays transcript results for a session in an expander.
    """
    expander = st.expander(f"Session {session_id}: Transcript Ready!")
    with expander:
        for update in poll_status(session_id, f"Session {session_id}"):
            if isinstance(update, pd.DataFrame):
                st.dataframe(update)
            else:
                st.write(update)
