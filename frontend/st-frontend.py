import streamlit as st
import requests

# Streamlit webpage layout
st.title('Audio Transcription Service')
st.write('Upload an audio file and select your preferences.')

# Sidebar for settings
with st.sidebar:
    st.header('Settings')
    size_of_model = st.selectbox("", ["small", "large"], index=1)
    task = st.selectbox("Task", ["transcribe", "translate"], index=0)  # Default to transcribe
    
    # Updated language selection with a dropdown
    languages = ['','english', 'chinese', 'german', 'spanish', 'russian', 'korean', 'french', 'japanese', 'portuguese', 'turkish', 'polish', 'catalan', 'dutch', 'arabic', 'swedish', 'italian', 'indonesian', 'hindi', 'finnish', 'vietnamese', 'hebrew', 'ukrainian', 'greek', 'malay', 'czech', 'romanian', 'danish', 'hungarian', 'tamil', 'norwegian', 'thai', 'urdu', 'croatian', 'bulgarian', 'lithuanian', 'latin', 'maori', 'malayalam', 'welsh', 'slovak', 'telugu', 'persian', 'latvian', 'bengali', 'serbian', 'azerbaijani', 'slovenian', 'kannada', 'estonian', 'macedonian', 'breton', 'basque', 'icelandic', 'armenian', 'nepali', 'mongolian', 'bosnian', 'kazakh', 'albanian', 'swahili', 'galician', 'marathi', 'punjabi', 'sinhala', 'khmer', 'shona', 'yoruba', 'somali', 'afrikaans', 'occitan', 'georgian', 'belarusian', 'tajik', 'sindhi', 'gujarati', 'amharic', 'yiddish', 'lao', 'uzbek', 'faroese', 'haitian creole', 'pashto', 'turkmen', 'nynorsk', 'maltese', 'sanskrit', 'luxembourgish', 'myanmar', 'tibetan', 'tagalog', 'malagasy', 'assamese', 'tatar', 'hawaiian', 'lingala', 'hausa', 'bashkir', 'javanese', 'sundanese', 'cantonese', 'burmese', 'valencian', 'flemish', 'haitian', 'letzeburgesch', 'pushto', 'panjabi', 'moldavian', 'moldovan', 'sinhalese', 'castilian', 'mandarin']
    source_language = st.selectbox("Source Language (blank for autodetection)", options=languages, index=0)  # Default to autodetect
    
    speaker_number = st.number_input("Speaker Number (0 for autodetection)", min_value=0, value=0, step=1)

# File uploader widget
uploaded_file = st.file_uploader("Choose an audio file...", type=['wav', 'mp3', 'mp4'])

# API Endpoint - Adjust with your actual FastAPI endpoint URL
API_ENDPOINT = 'http://localhost:8000/transcribe/'

if uploaded_file is not None:
    # Display a message to indicate that the file is being processed
    with st.spinner('Processing...'):
        # Convert the uploaded file to the appropriate format for sending
        files = {'file': (uploaded_file.name, uploaded_file, 'audio/*')}
        # Include the settings in the request
        data = {
            "size_of_model": size_of_model,
            "task": task,
            "source_language": source_language if source_language else "",  # Send empty string if autodetect
            "speaker_number": speaker_number
        }
        
        # POST request to send the file and settings to the backend API
        response = requests.post(API_ENDPOINT, files=files, data=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Display the transcription results
            transcription = response.json()
            st.success('Processing complete!')
            st.write(transcription)
        else:
            # Display an error message if something went wrong
            st.error('An error occurred while processing the file.')
