import os
import time
import requests
import yt_dlp
import pandas as pd
import tempfile
import streamlit as st

API_ENDPOINT = os.getenv('API_ENDPOINT', 'http://localhost:8000')

USE_MTLS = os.getenv('USE_MTLS', '0') == '1'


def secure_request(method, url, **kwargs):
    if USE_MTLS:
        client_cert = '/app/certs/signed_client.crt'
        client_key = '/app/certs/client.key'
        ca_cert = '/app/certs/ca.crt'
        kwargs['cert'] = (client_cert, client_key)
        kwargs['verify'] = ca_cert
    if method.lower() == 'get':
        return requests.get(url, **kwargs)
    elif method.lower() == 'post':
        return requests.post(url, **kwargs)
    raise ValueError("Method not supported")

# Load the supported languages from a file


def load_languages(file_path):
    with open(file_path, 'r') as file:
        languages = file.read().splitlines()
    return languages

# Function to download YouTube videos using yt_dlp


def download_youtube_video(url):
    """ Download YouTube video using yt_dlp with specific options. """
    ydl_opts = {
        'format': 'worst',  # Selects the worst available quality
        # Set download path to temp directory
        'paths': {'home': tempfile.gettempdir()},
        # Save the file with its title as the name
        'outtmpl': '%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        # Get the full path of the downloaded file
        video_file_path = ydl.prepare_filename(result)
        video_file_name = os.path.basename(
            video_file_path)  # Get just the file name

    return video_file_path, video_file_name  # Return both path and file name
# Process file uploads and returns session ids


def process_file(
        file_input,
        file_name,
        size_of_model,
        task_str,
        source_language,
        speaker_number):
    data = {
        "size_of_model": size_of_model,
        "task_str": task_str,
        "source_language": "" if source_language == '*Autodetect' else source_language,
        "speaker_number": "0" if speaker_number == '*Autodetect' else speaker_number,
    }
    files = {'file': (file_name, file_input, 'audio/*')}
    response = secure_request(
        'post',
        f"{API_ENDPOINT}/transcribe/",
        files=files,
        data=data)
    return handle_response(response, file_name)


def handle_response(response, file_name):
    if response.status_code == 200:
        json_response = response.json()
        session_id = json_response.get('session_id')
        return {
            'message': f'Successfully submitted. Your transcript will appear below shortly. Filename: {file_name} Task ID: {session_id}',
            'session_id': session_id}
    else:
        return {
            'error': f'Error submitting {file_name}: {response.status_code} - {response.text}',
            'session_id': None}


def poll_status(session_id, file_name):
    status = "queued"
    try:
        while status in ["queued", "processing"]:
            time.sleep(2)
            status_response = secure_request(
                'get', f"{API_ENDPOINT}/task_status/{session_id}")
            if status_response.status_code == 200:
                status_info = status_response.json()
                status = status_info.get('status')
                position = status_info.get('position')
                if position is not None:
                    yield f"{file_name} - Current queue position: {position}"
                if status == "completed":
                    if 'segments' in status_info:
                        transcription_df = pd.DataFrame(
                            status_info['segments'])
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
