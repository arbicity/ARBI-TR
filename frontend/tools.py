import os
import time
import requests
from pytube import YouTube
import pandas as pd
import tempfile
import streamlit as st

API_ENDPOINT = os.getenv('API_ENDPOINT', 'http://localhost:8000')

# Load the supported languages from a file
def load_languages(file_path):
    with open(file_path, 'r') as file:
        languages = file.read().splitlines()
    return languages

# Function to download YouTube videos using pytube
def download_youtube_video(url):
    yt = YouTube(url)
    video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().last()
    video_file_path = video.download(output_path=tempfile.gettempdir())
    return video_file_path

# Process file uploads and returns session ids
def process_file(file_input, file_name, size_of_model, task_str, source_language, speaker_number):
    data = {
        "size_of_model": size_of_model,
        "task_str": task_str,
        "source_language": "" if source_language == '*Autodetect' else source_language,
        "speaker_number": "0" if speaker_number == '*Autodetect' else speaker_number,
    }
    files = {'file': (file_name, file_input, 'audio/*')}
    response = requests.post(f"{API_ENDPOINT}/transcribe/", files=files, data=data)
    return handle_response(response, file_name)

# Handle API responses
def handle_response(response, file_name):
    if response.status_code == 200:
        json_response = response.json()
        session_id = json_response.get('session_id')
        return {'message': f'{file_name} submitted successfully. Task ID: {session_id}', 'session_id': session_id}
    else:
        return {'error': f'Error submitting {file_name}: {response.status_code} - {response.text}', 'session_id': None}

import requests

def poll_status(session_id, file_name):
    status = "queued"
    try:
        while status in ["queued", "processing"]:
            time.sleep(2)  # Poll every 2 seconds
            status_response = requests.get(f"{API_ENDPOINT}/task_status/{session_id}")
            if status_response.status_code == 200:
                status_info = status_response.json()
                status = status_info.get('status')
                position = status_info.get('position')

                if position is not None:
                    yield f"{file_name} - Current queue position: {position}"
                else:
                    yield f"{file_name} - Status: {status}"

                if status == "completed":
                    if 'segments' in status_info:
                        transcription_df = pd.DataFrame(status_info['segments'])
                        yield transcription_df
                    else:
                        yield f"Error: No transcription data found for {file_name}."
                    break
                elif status == "failed":
                    yield f'Processing failed for {file_name}: ' + status_info.get('error', 'Unknown error')
                    break
            else:
                yield f'Failed to fetch task status for {file_name}. Please try again later.'
                break
    except Exception as e:
        yield f'An error occurred while polling status: {str(e)}'

