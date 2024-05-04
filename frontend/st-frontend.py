import streamlit as st
import requests
import pandas as pd
import os
import time

# Streamlit UI setup
st.title('Audio Transcription Service')
st.write('Upload an audio file and select your preferences.')

# Definition of languages and speaker options
languages = [
    '*Autodetect', 'english', 'chinese', 'german', 'spanish', 'russian', 'korean',
    'french', 'japanese', 'portuguese', 'turkish', 'polish', 'catalan', 'dutch',
    'arabic', 'swedish', 'italian', 'indonesian', 'hindi', 'finnish', 'vietnamese',
    'hebrew', 'ukrainian', 'greek', 'malay', 'czech', 'romanian', 'danish', 'hungarian',
    'tamil', 'norwegian', 'thai', 'urdu', 'croatian', 'bulgarian', 'lithuanian', 'latin',
    'maori', 'malayalam', 'welsh', 'slovak', 'telugu', 'persian', 'latvian', 'bengali',
    'serbian', 'azerbaijani', 'slovenian', 'kannada', 'estonian', 'macedonian', 'breton',
    'basque', 'icelandic', 'armenian', 'nepali', 'mongolian', 'bosnian', 'kazakh', 'albanian',
    'swahili', 'galician', 'marathi', 'punjabi', 'sinhala', 'khmer', 'shona', 'yoruba', 'somali',
    'afrikaans', 'occitan', 'georgian', 'belarusian', 'tajik', 'sindhi', 'gujarati', 'amharic',
    'yiddish', 'lao', 'uzbek', 'faroese', 'haitian creole', 'pashto', 'turkmen', 'nynorsk',
    'maltese', 'sanskrit', 'luxembourgish', 'myanmar', 'tibetan', 'tagalog', 'malagasy', 'assamese',
    'tatar', 'hawaiian', 'lingala', 'hausa', 'bashkir', 'javanese', 'sundanese', 'cantonese',
    'burmese', 'valencian', 'flemish', 'haitian', 'letzeburgesch', 'pushto', 'panjabi', 'moldavian',
    'moldovan', 'sinhalese', 'castilian', 'mandarin'
]
speaker_options = ['*Autodetect', '1', '2', '3', '4', '5', '6', '7']

with st.sidebar:
    st.header('Settings')
    size_of_model = st.selectbox("Model Size", ["small", "large"], index=1)
    task = st.selectbox("Task", ["transcribe", "translate"], index=0)
    source_language = st.selectbox("Source Language", options=languages, index=1)
    speaker_number = st.selectbox("Number of Speakers", options=speaker_options, index=0)

uploaded_file = st.file_uploader("Choose an audio or video file...", type=['wav', 'mp3', 'mp4', 'm4a'])
API_ENDPOINT = os.getenv('API_ENDPOINT', 'https://arbi-tr-api.arbicity.com/transcribe/')

if uploaded_file is not None:
    files = {'file': (uploaded_file.name, uploaded_file, 'audio/*')}
    data = {
        "size_of_model": size_of_model,
        "task": task,
        "source_language": "" if source_language == '*Autodetect' else source_language,
        "speaker_number": 0 if speaker_number == '*Autodetect' else int(speaker_number)
    }

    response = requests.post(API_ENDPOINT, files=files, data=data)

    if response.status_code == 200:
        session_id = response.json().get('session_id')
        st.success('File uploaded successfully. Processing started...')
        progress_bar = st.progress(0)

        # Polling loop to check the transcription status
        complete = False
        progress = 0
        while not complete:
            time.sleep(1)  # Sleep time could be adjusted based on expected task duration
            status_response = requests.get(f"https://arbi-tr-api.arbicity.com/task_status/{session_id}")
            if status_response.status_code == 200:
                status_info = status_response.json()
                if status_info['status'] == 'completed':
                    transcription_df = pd.DataFrame(status_info['data']['segments'])
                    st.dataframe(transcription_df)  # Corrected method to display DataFrame
                    complete = True
                elif status_info['status'] == 'failed':
                    st.error('Transcription failed: ' + status_info.get('error', 'Unknown error'))
                    complete = True
            else:
                st.error('Failed to fetch status. Please try again.')
                complete = True
            progress += 10
            progress_bar.progress(min(progress, 100))
    else:
        st.error(f'An error occurred while processing the file: {response.status_code} - {response.text}')
