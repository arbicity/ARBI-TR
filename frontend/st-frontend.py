import streamlit as st
import requests
import pandas as pd

# Streamlit UI
st.title('Audio Transcription Service')
st.write('Upload an audio file and select your preferences.')

# Sidebar for settings
with st.sidebar:
    st.header('Settings')
    size_of_model = st.selectbox("Model Size", ["small", "large"], index=1)
    task = st.selectbox("Task", ["transcribe", "translate"], index=0)
    languages = ['*Autodetect', 'english', 'chinese', 'german', 'spanish', 'russian', 'korean', 'french', 'japanese', 'portuguese', 'turkish', 'polish', 'catalan', 'dutch', 'arabic', 'swedish', 'italian', 'indonesian', 'hindi', 'finnish', 'vietnamese', 'hebrew', 'ukrainian', 'greek', 'malay', 'czech', 'romanian', 'danish', 'hungarian', 'tamil', 'norwegian', 'thai', 'urdu', 'croatian', 'bulgarian', 'lithuanian', 'latin', 'maori', 'malayalam', 'welsh', 'slovak', 'telugu', 'persian', 'latvian', 'bengali', 'serbian', 'azerbaijani', 'slovenian', 'kannada', 'estonian', 'macedonian', 'breton', 'basque', 'icelandic', 'armenian', 'nepali', 'mongolian', 'bosnian', 'kazakh', 'albanian', 'swahili', 'galician', 'marathi', 'punjabi', 'sinhala', 'khmer', 'shona', 'yoruba', 'somali', 'afrikaans', 'occitan', 'georgian', 'belarusian', 'tajik', 'sindhi', 'gujarati', 'amharic', 'yiddish', 'lao', 'uzbek', 'faroese', 'haitian creole', 'pashto', 'turkmen', 'nynorsk', 'maltese', 'sanskrit', 'luxembourgish', 'myanmar', 'tibetan', 'tagalog', 'malagasy', 'assamese', 'tatar', 'hawaiian', 'lingala', 'hausa', 'bashkir', 'javanese', 'sundanese', 'cantonese', 'burmese', 'valencian', 'flemish', 'haitian', 'letzeburgesch', 'pushto', 'panjabi', 'moldavian', 'moldovan', 'sinhalese', 'castilian', 'mandarin']
    source_language = st.selectbox("Source Language", options=languages, index=1)  # Set English as default, index adjusted for English
    speaker_options = ['*Autodetect', '1', '2', '3', '4', '5', '6', '7']
    speaker_number = st.selectbox("Number of Speakers", options=speaker_options, index=0)  # Use selectbox for speaker number with autodetect as default

uploaded_file = st.file_uploader("Choose an audio or video file...", type=['wav', 'mp3', 'mp4', 'm4a'])

API_ENDPOINT = 'http://arbi-tr-api-service:8000/transcribe/'

if uploaded_file is not None:
    with st.spinner('Processing...'):
        files = {'file': (uploaded_file.name, uploaded_file, 'audio/*')}
        data = {
            "size_of_model": size_of_model,
            "task": task,
            "source_language": "" if source_language == '*Autodetect' else source_language,
            "speaker_number": 0 if speaker_number == '*Autodetect' else speaker_number
        }
        
        response = requests.post(API_ENDPOINT, files=files, data=data)
        
        if response.status_code == 200:
            st.success('Processing complete!')
            transcription_df = pd.DataFrame(response.json()['segments'])
            st.data_editor(transcription_df, num_rows="dynamic")
        else:
            st.error('An error occurred while processing the file.')
