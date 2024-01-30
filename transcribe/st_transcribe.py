import os

import streamlit as st

from transcribe import get_youtube, speech_to_text

st.title("Transcribe a recording")

# Initialize session state variables if they don't exist
if "media_file_path" not in st.session_state:
    st.session_state.media_file_path = None
if "file_ready_for_transcription" not in st.session_state:
    st.session_state.file_ready_for_transcription = False

# Sidebar options
st.sidebar.subheader("Transcription Options")
selected_source_lang = st.sidebar.selectbox(
    "Spoken language in recording:", ["en", "jp", "de", "fr", "es", "..."]
)  # Add languages as needed
whisper_models = {"fastest": "tiny", "balanced": "small", "most accurate": "large-v2"}
whisper_model = st.sidebar.radio("Whisper Model:", list(whisper_models.keys()))
whisper_model = whisper_models[whisper_model]
num_speakers = st.sidebar.slider("Number of speakers (set to 0 to auto-detect):", 0, 10, 0)

# Upload the video
media_file = st.file_uploader("Choose an input file", type=["mp4", "mp3", "m4a", "mov", "wmv", "wav"])

# Add YouTube URL input
st.subheader("Or, enter a YouTube URL:")
youtube_url = st.text_input("YouTube URL")
download_youtube = st.button("Download from YouTube")

# Handle YouTube download
if download_youtube and youtube_url:
    try:
        st.session_state.media_file_path = get_youtube(youtube_url)
        st.session_state.file_ready_for_transcription = True
        st.success(f"Successfully downloaded video: {os.path.basename(st.session_state.media_file_path)}")
        st.video(st.session_state.media_file_path)
    except Exception as e:
        st.error(f"Failed to download video: {e}")
        st.session_state.file_ready_for_transcription = False

# Display the uploaded video or audio file
if media_file:
    st.session_state.media_file_path = os.path.join("uploaded_files", media_file.name)
    with open(st.session_state.media_file_path, "wb") as f:
        f.write(media_file.getvalue())
    st.session_state.file_ready_for_transcription = True
    file_type = media_file.type.split("/")[0]
    if file_type == "audio":
        st.audio(media_file, format="audio/" + media_file.type.split("/")[1])
    elif file_type == "video":
        st.video(media_file, format="video/" + media_file.type.split("/")[1])

# Transcribe
if st.button("Transcribe"):
    if st.session_state.file_ready_for_transcription:
        df_results, system_info, _ = speech_to_text(
            st.session_state.media_file_path,
            selected_source_lang,
            whisper_model,
            num_speakers,
        )

        st.subheader("Transcription Results")
        # Using text_area instead of write for a scrollable container with word wrap
        edited_df = st.data_editor(df_results, height=300)
        st.write(system_info)

        # Offer to download as CSV
        csv_data = df_results.to_csv(index=False)
        st.download_button(
            label="Save as CSV",
            data=csv_data,
            file_name="transcription.csv",
            mime="text/csv",
        )
    else:
        st.warning("Please upload a file or enter a valid YouTube URL first.")

# This app will save the uploaded recording in the "uploaded_files" directory, which should exist.
os.makedirs("uploaded_files", exist_ok=True)
