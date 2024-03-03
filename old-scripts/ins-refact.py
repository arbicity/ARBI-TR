import contextlib
import csv
import datetime
import os
import subprocess
import time
from typing import Tuple, List

import numpy as np
import torch
import torchaudio
from sklearn.cluster import AgglomerativeClustering
from transformers import pipeline, utils
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

def convert_time(secs: float) -> str:
    """Converts seconds to a HH:MM:SS format string for easier readability."""
    return str(datetime.timedelta(seconds=round(secs)))

def timeit(func):
    """Decorator to measure the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} executed in {time.time() - start_time:.4f} seconds")
        return result
    return wrapper

@timeit
def convert_audio_to_wav(media_file_path: str) -> str:
    """Converts an audio file to WAV format using ffmpeg."""
    base_name, _ = os.path.splitext(media_file_path)
    audio_file = f"{base_name}(converted).wav"
    if not os.path.exists(audio_file):
        cmd = ['ffmpeg', '-i', media_file_path, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', audio_file]
        subprocess.run(cmd, check=True)
        print(f"Converted {media_file_path} to WAV.")
    else:
        print("WAV file already exists.")
    return audio_file

@timeit
def load_audio_file(file_path: str) -> Tuple[torch.Tensor, int]:
    """Loads an audio file into memory."""
    return torchaudio.load(file_path)

@timeit
def generate_embeddings_for_segments(embedding_model, waveform: torch.Tensor, sample_rate: int, segments: List[Tuple[float, float]]) -> List[torch.Tensor]:
    """Generates embeddings for a list of given audio segments."""
    embeddings = []
    for start_time, end_time in segments:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample].unsqueeze(0)
        embedding = embedding_model(segment_waveform)
        embeddings.append(embedding)
    return embeddings

@timeit
def perform_transcription(asr_pipeline, converted_audio_file_path: str):
    """Performs transcription using the specified ASR pipeline."""
    return asr_pipeline(converted_audio_file_path, chunk_length_s=30, batch_size=48, return_timestamps=True)

@timeit
def generate_transcription_output(segments, speaker_labels) -> dict:
    """Generates transcription output with speaker labels."""
    objects = {'Start': [], 'End': [], 'Speaker': [], 'Text': []}
    text, current_speaker = '', None
    for i, segment in enumerate(segments):
        start_time, end_time = segment["timestamp"]
        speaker = speaker_labels[i]
        if i == 0 or current_speaker != speaker:
            if i > 0:
                objects['End'].append(convert_time(segments[i - 1]["timestamp"][1]))
                objects['Text'].append(text.strip())
                text = ''
            objects['Start'].append(convert_time(start_time))
            current_speaker = speaker
            objects['Speaker'].append(str(current_speaker))
        text += segment["text"] + ' '
    if segments:
        objects['End'].append(convert_time(segments[-1]["timestamp"][1]))
        objects['Text'].append(text.strip())
    return objects

def cluster_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """Clusters embeddings and returns the labels."""
    embeddings_array = np.vstack(embeddings)
    clustering_model = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=2000)
    clustering_model.fit(embeddings_array)
    print(f"Clustering completed. Labels: {np.unique(clustering_model.labels_)}")
    return clustering_model.labels_

def save_to_csv(data: dict, file_path: str):
    """Saves the transcription and speaker labels to a CSV file."""
    with open(file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Start Time', 'End Time', 'Speaker', 'Transcript'])
        for start_time, end_time, speaker, text in zip(data['Start'], data['End'], data['Speaker'], data['Text']):
            csv_writer.writerow([start_time, end_time, speaker, text])
    print(f"Transcript with speaker labels saved to {file_path}")

def main():
    """Main function to orchestrate the conversion, transcription, and analysis process."""
    # Initialize the ASR pipeline
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16,
        model_kwargs={
            "device_map": "cuda:0",
            "attn_implementation": "sdpa"
        },
        generate_kwargs={
            "task": "transcribe",  # can be either "transcribe" or "translate"
            "language": ""  # specifies source language if known
        }
    )    
    # Convert the audio file to WAV format
    converted_audio_file_path = convert_audio_to_wav("samples/svamc.wav")
    
    # Load the entire audio file into memory
    waveform, sample_rate = load_audio_file(converted_audio_file_path)
    
    # Initialize the speaker embedding model
    embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device="cuda")
    
    # Perform transcription and get result
    transcription_result = perform_transcription(asr_pipeline, converted_audio_file_path)
    
    # Prepare segment times for embedding generation
    segments = [(chunk["timestamp"][0], chunk["timestamp"][1]) for chunk in transcription_result["chunks"]]
    
    # Generate embeddings for all segments at once
    embeddings = generate_embeddings_for_segments(embedding_model, waveform, sample_rate, segments)
    
    # Proceed with clustering and generating speaker labels
    speaker_labels = cluster_embeddings(embeddings) if len(embeddings) > 1 else np.zeros(len(embeddings), dtype=int)
    
    # Generate transcription output with speaker labels
    grouped_segments = generate_transcription_output(transcription_result["chunks"], speaker_labels)
    
    # Save the results to a CSV file
    save_to_csv(grouped_segments, "transcript_with_speaker_labels.csv")

# Ensure the generate_embeddings_for_segments is properly defined as discussed earlier

if __name__ == "__main__":
    main()
