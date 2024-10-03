# utils.py
import subprocess
import tempfile
from itertools import islice
import datetime
import os
import time
import torch
import torchaudio
from sklearn.cluster import AgglomerativeClustering
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import numpy as np
from typing import Tuple, List, Optional
import gc
import logging

# Configure logging at the beginning of your script

def convert_time(secs: float) -> str:
    if secs is None:
        return "00:00:00"  # Handle None values gracefully
    return str(datetime.timedelta(seconds=round(secs)))


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} executed in {time.time() - start_time:.4f} seconds")
        return result
    return wrapper

@timeit
def convert_audio_to_wav(media_file_path: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        audio_file_path = temp_wav_file.name
    try:
        # Command setup, ensure the command ends properly with the file path arguments
        cmd = ['ffmpeg', '-y', '-i', media_file_path, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', '-preset', 'ultrafast', audio_file_path]
        subprocess.run(cmd, check=True)
        print(f"Converted {media_file_path} to WAV: {audio_file_path}")
    except Exception as e:
        os.remove(audio_file_path)
        raise e
    return audio_file_path

@timeit
def load_audio_file(file_path: str) -> Tuple[torch.Tensor, int]:
    return torchaudio.load(file_path)

@timeit
def perform_transcription(asr_pipeline, converted_audio_file_path: str):
    return asr_pipeline(converted_audio_file_path, chunk_length_s=30, batch_size=24, return_timestamps=True)

@timeit
def generate_transcription_output(segments, speaker_labels) -> dict:
    output_segments = []
    text, current_speaker, prev_start_time, prev_end_time = '', None, '', ''
    for i, segment in enumerate(segments):
        start_time = convert_time(segment["timestamp"][0])
        end_time = convert_time(segment["timestamp"][1])
        speaker = speaker_labels[i]

        if current_speaker is None or current_speaker != speaker:
            if text:
                output_segments.append({
                    "Start": prev_start_time,
                    "End": prev_end_time,
                    "Speaker": str(current_speaker),
                    "Text": text.strip()
                })
                text = ''
            current_speaker = speaker
            prev_start_time = start_time
        text += segment["text"] + ' '
        prev_end_time = end_time

    if text:
        output_segments.append({
            "Start": prev_start_time,
            "End": prev_end_time,
            "Speaker": str(current_speaker),
            "Text": text.strip()
        })

    return {"segments": output_segments}

@timeit
def generate_embeddings_for_segments(
    embedding_model, 
    waveform: torch.Tensor, 
    sample_rate: int, 
    segments: List[Tuple[float, float]]
) -> List[np.ndarray]:
    """
    Generate speaker embeddings for audio segments with memory conservation using torch.no_grad().
    
    Args:
        embedding_model: The pretrained speaker embedding model.
        waveform (torch.Tensor): The audio waveform tensor.
        sample_rate (int): The sampling rate of the audio.
        segments (List[Tuple[float, float]]): List of (start_time, end_time) tuples.
    
    Returns:
        List[np.ndarray]: List of embedding vectors for each segment.
    """
    embeddings = []
    audio_length_in_seconds = waveform.shape[1] / sample_rate
    EMBEDDING_DIM = 192  # Replace with the correct dimension if different

    # Disable gradient calculations to save memory
    with torch.no_grad():
        for idx, (start_time, end_time) in enumerate(segments):
            # Ensure end_time does not exceed the audio length
            if end_time is None or end_time > audio_length_in_seconds:
                end_time = audio_length_in_seconds

            # Optional: Limit maximum segment duration to prevent very long segments
            MAX_SEGMENT_DURATION = 300  # in seconds (e.g., 5 minutes)
            end_time = min(end_time, start_time + MAX_SEGMENT_DURATION, audio_length_in_seconds)

            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_waveform = waveform[:, start_sample:end_sample].unsqueeze(0).to(embedding_model.device)

            try:
                # Generate embedding for the segment
                embedding = embedding_model(segment_waveform)
                
                if isinstance(embedding, np.ndarray):
                    embeddings.append(embedding)
                else:
                    # If embedding is a PyTorch tensor, move it to CPU and convert to NumPy
                    embeddings.append(embedding.cpu().numpy())
            except Exception as e:
                logging.error(f"Error generating embedding for segment {idx}: {e}")
                # Append a zero vector as a placeholder to maintain alignment
                embeddings.append(np.zeros(EMBEDDING_DIM))
                continue

            # Clean up memory by deleting variables
            del segment_waveform, embedding

            # Invoke garbage collection periodically
            if (idx + 1) % 500 == 0:
                gc.collect()
                logging.info(f"Processed {idx + 1} segments and invoked garbage collection.")

            # Clear CUDA cache periodically to free up GPU memory
            if (idx + 1) % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logging.info(f"Processed {idx + 1} segments.")

    return embeddings

@timeit
def cluster_embeddings(embeddings: List[torch.Tensor]) -> np.ndarray:
    embeddings_array = np.vstack(embeddings)
    embeddings_with_nans = set()  # Keep track of indices of embeddings with NaN values

    # Check for NaN values in the embeddings array
    if np.isnan(embeddings_array).any():
        # Impute NaN values using the mean of columns
        for i in range(embeddings_array.shape[1]):
            column_mean = np.nanmean(embeddings_array[:, i])
            nan_indices = np.where(np.isnan(embeddings_array[:, i]))[0]
            embeddings_array[nan_indices, i] = column_mean
            # Add indices of embeddings with NaN values to the set
            embeddings_with_nans.update(nan_indices)

    # Perform clustering with the cleaned embeddings array
    clustering_model = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=1200)
    clustering_model.fit(embeddings_array)
    print(f"Clustering completed. Labels: {np.unique(clustering_model.labels_)}")

    # Output the indices of embeddings with NaN values
    for idx in embeddings_with_nans:
        print(f"Warning: Embedding at index {idx} contained NaN values which were replaced by mean imputed values.")
    return clustering_model.labels_

@timeit
def process_audio(file_path: str, size_of_model: str, task: str, source_language: Optional[str], speaker_number: Optional[int]):
    converted_audio_file_path = convert_audio_to_wav(file_path)
    logging.basicConfig(level=logging.INFO)

    model_id = f"openai/whisper-{size_of_model}" + ("-v3-turbo" if size_of_model == "large" else "")

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch.float16,
        model_kwargs={
            "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
            "attn_implementation": "sdpa"
        },
        generate_kwargs={
            "task": task,
            "language": source_language
        }
    )

    waveform, sample_rate = load_audio_file(converted_audio_file_path)
    transcription_result = perform_transcription(asr_pipeline, converted_audio_file_path)
    del asr_pipeline  # Clean up the ASR model resources

    valid_chunks = [chunk for chunk in transcription_result["chunks"] if chunk["timestamp"][1] is not None]
    segments = [(chunk["timestamp"][0], chunk["timestamp"][1]) for chunk in valid_chunks]

    # Check if there are enough segments for clustering
    if len(segments) < 2:
        print("Warning: Only 1 segment found - either a very short file or could not be processed correctly.")
        # Simulate a grouped segment with only one speaker and one segment
        if valid_chunks:
            grouped_segments = generate_transcription_output(valid_chunks, [0] * len(valid_chunks))  # All segments attributed to a single speaker
        else:
            grouped_segments = {"segments": []}
        os.remove(file_path)
        os.remove(converted_audio_file_path)
        return grouped_segments

    embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device="cuda")
    embeddings = generate_embeddings_for_segments(embedding_model, waveform, sample_rate, segments)

    if speaker_number == 0:
        speaker_labels = cluster_embeddings(embeddings)
    else:
        clustering_model = AgglomerativeClustering(n_clusters=speaker_number)
        speaker_labels = clustering_model.fit_predict(np.vstack(embeddings))

    grouped_segments = generate_transcription_output(valid_chunks, speaker_labels)

    del embedding_model  # Clean up the embedding model resources

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.remove(file_path)
    os.remove(converted_audio_file_path)

    return grouped_segments

@timeit
def process_audio_without_diarization(file_path: str, size_of_model: str, task: str, source_language: Optional[str]):
    converted_audio_file_path = convert_audio_to_wav(file_path)

    model_id = f"openai/whisper-{size_of_model}" + ("-v3" if size_of_model == "large" else "")

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch.float16,
        model_kwargs={
            "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
            "attn_implementation": "sdpa"
        },
        generate_kwargs={
            "task": task,
            "language": source_language
        }
    )

    transcription_result = perform_transcription(asr_pipeline, converted_audio_file_path)
    del asr_pipeline  # Clean up the ASR model resources

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.remove(converted_audio_file_path)

    return {"text": transcription_result["text"]}
