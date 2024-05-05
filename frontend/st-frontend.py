import streamlit as st
import requests
import pandas as pd
import os
import time

# Streamlit UI setup
st.title('Audio Transcription Service')
st.write('Upload audio files and select your preferences.')

def load_languages(file_path):
    with open(file_path, 'r') as file:
        languages = file.read().splitlines()
    return languages

# Definition of languages and speaker options
languages_file_path = 'languages.txt'  # Adjust if your file is in a different location
languages = load_languages(languages_file_path)
speaker_options = ['*Autodetect', '1', '2', '3', '4', '5', '6', '7']

with st.sidebar:
    st.header('Settings')
    size_of_model = st.selectbox("Model Size", ["small", "large"], index=1)
    task_str = st.selectbox("Task", ["transcribe", "translate"], index=0)
    source_language = st.selectbox("Source Language", options=languages, index=languages.index('*Autodetect'))
    speaker_number = st.selectbox("Number of Speakers", options=speaker_options, index=0)

uploaded_files = st.file_uploader("Choose audio or video files...", type=['wav', 'mp3', 'mp4', 'm4a'], accept_multiple_files=True)

API_ENDPOINT = os.getenv('API_ENDPOINT', 'https://arbi-tr-api.arbicity.com')

def process_file(uploaded_file):
    st.write(f"Uploading {uploaded_file.name}...")

    # Prepare and send the request
    files = {'file': (uploaded_file.name, uploaded_file, 'audio/*')}
    data = {
        "size_of_model": size_of_model,
        "task_str": task_str,
        "source_language": "" if source_language == '*Autodetect' else source_language,
        "speaker_number": "0" if speaker_number == '*Autodetect' else speaker_number,
    }

    response = requests.post(f"{API_ENDPOINT}/transcribe/", files=files, data=data)

    if response.status_code == 200:
        session_id = response.json().get('session_id')
        st.success(f'{uploaded_file.name} uploaded successfully. Processing started, session ID: {session_id}')

        # Poll for task status
        return session_id
    else:
        st.error(f'Error uploading {uploaded_file.name}: {response.status_code} - {response.text}')
        return None


def poll_status(session_id, file_name):
    status = "queued"
    while status in ["queued", "processing"]:
        time.sleep(2)  # Poll every 2 seconds to check the status
        status_response = requests.get(f"{API_ENDPOINT}/task_status/{session_id}")

        if status_response.status_code == 200:
            status_info = status_response.json()
            status = status_info.get('status')
            position = status_info.get('position')

            # Display current queue position if available
            if position is not None:
                st.write(f"{file_name} - Current queue position: {position}")
            else:
                st.write(f"{file_name} - Status: {status}")

            # Check if the task has completed and handle accordingly
            if status == "completed":
                if 'segments' in status_info:
                    transcription_df = pd.DataFrame(status_info['segments'])
                    st.write(f"{file_name} - Processing completed successfully.")
                    st.dataframe(transcription_df)
                else:
                    st.error(f"No transcription data found for {file_name}. Check backend logs for more details.")
                break  # Exit the loop since processing is complete
            elif status == "failed":
                st.error(f'Processing failed for {file_name}: ' + status_info.get('error', 'Unknown error'))
                break  # Exit the loop since processing has failed
        else:
            st.error(f'Failed to fetch task status for {file_name}. Please try again later.')
            break  # Exit the loop on failure to fetch status
if uploaded_files:
    session_ids = [process_file(file) for file in uploaded_files]
    for idx, session_id in enumerate(session_ids):
        if session_id:
            poll_status(session_id, uploaded_files[idx].name)
