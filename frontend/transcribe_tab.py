import streamlit as st
from transcribe_tools import load_languages, download_youtube_video, process_file, poll_status
import os
import pandas as pd

def transcribe_tab():
    col1, col2 = st.columns([.2, .8])
    with col1:
        st.image("ARBI_Assistant.png", width=75)
    with col2:
        st.title('Transcribe and Translate')
    with st.expander(label="About this tool"):
        st.markdown(open("about.md").read())

    st.write("Step 1.")
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_files = st.file_uploader("Upload audio/video file(s)", type=['wav', 'mp3', 'mp4', 'm4a'], accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                display_file = file.getvalue()
                _, file_ext = os.path.splitext(file.name)
                if file_ext in ['.mp4', '.m4a']:
                    st.video(display_file)
                else:
                    st.audio(display_file)

    with col2:
        youtube_url = st.text_area("or enter YouTube Video URLs, one per line")
        if 'downloaded_files' not in st.session_state:
            st.session_state['downloaded_files'] = []

        if youtube_url:
            urls = youtube_url.split('\n')
            for url in urls:
                if url.strip() and not any(url == d[2] for d in st.session_state['downloaded_files']):
                    with st.spinner(f'Downloading YouTube video from {url}...'):
                        try:
                            video_file_path, video_file_name = download_youtube_video(url)
                            st.session_state['downloaded_files'].append((video_file_path, video_file_name, url))
                            _, file_ext = os.path.splitext(video_file_path)
                            if file_ext in ['.mp4', '.m4a']:
                                st.video(open(video_file_path, 'rb'))
                            else:
                                st.audio(open(video_file_path, 'rb'))
                        except Exception as e:
                            st.error(f"Failed to download video from {url}. Error: {e}")

    st.write("Step 2.")
    with st.expander("Settings"):
        languages_file_path = 'languages.txt'
        languages = load_languages(languages_file_path)
        speaker_options = ['*Autodetect', '1', '2', '3', '4', '5', '6', '7', '8']
        size_of_model = st.selectbox("Speech recognition model size (small is faster, large is more accurate)", ["small", "large"], index=1)
        task_str = st.selectbox("Task", ["transcribe", "translate"], index=0)
        default_language_index = languages.index('english') if 'english' in languages else 0
        source_language = st.selectbox("Source Language", options=languages, index=default_language_index)
        speaker_number = st.selectbox("Number of Speakers", options=speaker_options, index=0)

    if st.button('Generate Transcript'):
        session_ids = []
        if uploaded_files:
            for file in uploaded_files:
                result = process_file(file.getvalue(), file.name, size_of_model, task_str, source_language, speaker_number)
                session_id = result.get('session_id')
                if session_id:
                    session_ids.append(session_id)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success(result['message'])
                    expander = st.expander("Done! Click to view and download the transcript")
                    with expander:
                        for update in poll_status(session_id, f"Session {session_id}"):
                            if isinstance(update, pd.DataFrame):
                                st.dataframe(update)
                            else:
                                st.write(update)

        for video_file_path, video_file_name, _ in st.session_state['downloaded_files']:
            result = process_file(open(video_file_path, 'rb'), video_file_name, size_of_model, task_str, source_language, speaker_number)
            session_id = result.get('session_id')
            if session_id:
                session_ids.append(session_id)
            if 'error' in result:
                st.error(result['error'])
            else:
                st.success(result['message'])
                expander = st.expander(f"Transcript Ready! Click here to view and download as a table")
                with expander:
                    for update in poll_status(session_id, f"Session {session_id}"):
                        if isinstance(update, pd.DataFrame):
                            st.dataframe(update)
                        else:
                            st.write(update)