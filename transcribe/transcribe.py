import contextlib
import datetime
import os
import subprocess
import time
import wave

import numpy as np
import pandas as pd
import psutil
import streamlit as st
import torch
import yt_dlp
from faster_whisper import WhisperModel
from gpuinfo import GPUInfo
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

os.makedirs("output", exist_ok=True)


def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))


def get_youtube(video_url):
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        abs_video_path = ydl.prepare_filename(info)
        ydl.process_info(info)

    print("Success download video")
    print(abs_video_path)
    return abs_video_path


def convert_audio_to_wav(media_file_path):
    base_name, file_ending = os.path.splitext(media_file_path)
    audio_file = f"{base_name}(converted).wav"

    # Check if (converted).wav file already exists
    if not os.path.exists(audio_file):
        cmd = ["ffmpeg", "-i", media_file_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_file]
        subprocess.run(cmd, check=True)

    return audio_file


def get_audio_duration(audio_file):
    with contextlib.closing(wave.open(audio_file, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration


def transcribe_audio(model, audio_file, selected_source_lang):
    options = dict(language=selected_source_lang, beam_size=5, best_of=5)
    transcribe_options = dict(task="transcribe", **options)
    segments_raw, info = model.transcribe(audio_file, **transcribe_options)
    segments = [dict(start=s.start, end=s.end, text=s.text) for s in segments_raw]
    return segments


def segment_embedding(segment, audio_file, duration):
    audio = Audio()
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(audio_file, clip)
    return embedding_model(waveform[None])


def compute_speaker_embedding(segments, audio_file, duration):
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment, audio_file, duration)
    return embeddings


def determine_best_speaker_number(embeddings, num_speakers):
    if num_speakers == 0:
        score_num_speakers = {}
        max_possible_clusters = len(embeddings)
        for i, potential_speaker_count in enumerate(range(2, min(10 + 1, max_possible_clusters))):
            try:
                clustering = AgglomerativeClustering(potential_speaker_count).fit(embeddings)

                # Avoid singleton clusters
                unique_labels, counts = np.unique(clustering.labels_, return_counts=True)
                if any(counts == 1):
                    continue

                score = silhouette_score(embeddings, clustering.labels_, metric="euclidean")
                score_num_speakers[potential_speaker_count] = score

            except ValueError as e:
                if "minimum of 2 is required" in str(e) and len(embeddings) == 1:
                    return 1  # Only one sample exists, return 1 speaker
                else:
                    # Some other error occurred, so re-raise the error to handle it elsewhere or present it to the user
                    raise e

        best_num_speaker = max(
            score_num_speakers, key=lambda x: score_num_speakers[x], default=2
        )  # added default to avoid potential issue if score_num_speakers is empty
    else:
        best_num_speaker = num_speakers

    return best_num_speaker


def assign_speaker_label(segments, best_num_speaker, embeddings):
    # Check if there's only one sample in embeddings
    if len(embeddings) == 1:
        # Since there's only one speaker, you can assign all segments to that speaker
        for segment in segments:
            segment["speaker"] = 1
        return

    try:
        clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
    except ValueError as e:
        if "minimum of 2 is required" in str(e):
            # Only one sample exists, return 1 speaker
            # Assign all segments to that speaker and exit the function
            for segment in segments:
                segment["speaker"] = 1
            return
        else:
            # Some other error occurred, so re-raise the error
            raise e

    for segment, label in zip(segments, clustering.labels_):
        segment["speaker"] = label + 1


def generate_transcription_output(segments):
    objects = {"Start": [], "End": [], "Speaker": [], "Text": []}
    text = ""
    for i, segment in enumerate(segments):
        if i == 0:
            # Start new segment for the first iteration
            objects["Start"].append(str(convert_time(segment["start"])))
            objects["Speaker"].append(segment["speaker"])
        elif segments[i - 1]["speaker"] != segment["speaker"]:
            # End the previous segment and start a new segment when the speaker changes
            objects["End"].append(str(convert_time(segments[i - 1]["end"])))
            objects["Text"].append(text.strip())

            objects["Start"].append(str(convert_time(segment["start"])))
            objects["Speaker"].append(segment["speaker"])
            text = ""

        text += segment["text"] + " "

    # Append the details of the last segment
    objects["End"].append(str(convert_time(segments[-1]["end"])))
    objects["Text"].append(text.strip())

    return objects


def get_system_info(start_time):
    time_end = time.time()
    time_diff = time_end - start_time
    memory = psutil.virtual_memory()

    gpu_utilization, gpu_memory = GPUInfo.gpu_usage()
    gpu_utilization = gpu_utilization[0] if len(gpu_utilization) > 0 else 0
    gpu_memory = gpu_memory[0] if len(gpu_memory) > 0 else 0

    system_info = {
        "Memory Total (GB)": memory.total / (1024 * 1024 * 1024),
        "Memory Used (%)": memory.percent,
        "Memory Available (GB)": memory.available / (1024 * 1024 * 1024),
        "Processing Time (seconds)": time_diff,
        "GPU Utilization (%)": gpu_utilization,
        "GPU Memory (MiB)": gpu_memory,
    }
    return system_info


def speech_to_text(media_file_path, selected_source_lang, whisper_model, num_speakers):
    # Progress bar initialization
    main_progress_bar = st.progress(0)
    start_time = time.time()

    # Convert media to WAV
    with st.spinner("Converting file format..."):
        audio_file = convert_audio_to_wav(media_file_path)
        main_progress_bar.progress(0.20)

    # Calculate audio duration
    duration = get_audio_duration(audio_file)

    # Transcription
    with st.spinner("Transcribing audio..."):
        segments = transcribe_audio(WhisperModel(whisper_model, compute_type="int8"), audio_file, selected_source_lang)
        main_progress_bar.progress(0.50)

    # Compute Speaker Embeddings
    with st.spinner("Creating embeddings..."):
        embeddings = compute_speaker_embedding(segments, audio_file, duration)
        main_progress_bar.progress(0.70)

    # Determine number of speakers
    best_num_speaker = determine_best_speaker_number(embeddings, num_speakers)

    # Assign Speaker Labels
    with st.spinner("Assigning speaker labels..."):
        assign_speaker_label(segments, best_num_speaker, embeddings)
        main_progress_bar.progress(0.85)

    # Generate final transcription output
    with st.spinner("Generating transcription output..."):
        objects = generate_transcription_output(segments)
        main_progress_bar.progress(0.95)

    # System info stats
    system_info = get_system_info(start_time)
    system_info_text = f"""
    Processing Time: {system_info["Processing Time (seconds)"]:.5} seconds
    GPU Utilization: {system_info["GPU Utilization (%)"]}%
    GPU Memory: {system_info["GPU Memory (MiB)"]} MiB
    """

    st.write(system_info_text)
    main_progress_bar.progress(1.0)

    save_path = "data/output/transcript_result.csv"
    df_results = pd.DataFrame(objects)
    df_results.to_csv(save_path)
    return df_results, system_info, save_path
