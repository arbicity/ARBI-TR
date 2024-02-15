from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil
import subprocess
import tempfile
import datetime
import os
import time
from typing import Tuple, List, Optional
import numpy as np
import torch
import torchaudio
from sklearn.cluster import AgglomerativeClustering
from transformers import pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import multiprocessing
from functools import partial

app = FastAPI()

def convert_time(secs: float) -> str:
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
        cmd = ['ffmpeg', '-y', '-i', media_file_path, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', audio_file_path]
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
    """Performs transcription using the specified ASR pipeline."""
    return asr_pipeline(converted_audio_file_path, chunk_length_s=30, batch_size=48, return_timestamps=True)

@timeit
def generate_transcription_output(segments, speaker_labels) -> dict:
    """Generates transcription output with speaker labels, reorganizing for grouped output."""
    output_segments = []
    text, current_speaker = '', None
    for i, segment in enumerate(segments):
        start_time = convert_time(segment["timestamp"][0])
        end_time = convert_time(segment["timestamp"][1])
        speaker = speaker_labels[i]

        if current_speaker is None or current_speaker != speaker:
            if text:  # If there's accumulated text, append the previous segment before starting a new one
                output_segments.append({
                    "Start": prev_start_time,
                    "End": prev_end_time,
                    "Speaker": str(current_speaker),
                    "Text": text.strip()
                })
                text = ''  # Reset text for the new segment

            current_speaker = speaker  # Update the current speaker
            prev_start_time = start_time  # Update start time for potential new segment

        text += segment["text"] + ' '
        prev_end_time = end_time  # Always update end time to the latest for ongoing speaker segments

    # After loop, add the last accumulated segment
    if text:
        output_segments.append({
            "Start": prev_start_time,
            "End": prev_end_time,
            "Speaker": str(current_speaker),
            "Text": text.strip()
        })

    return {"segments": output_segments}

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

def cluster_embeddings(embeddings: List[torch.Tensor]) -> np.ndarray:
    """Clusters embeddings and returns the labels."""
    embeddings_array = np.vstack(embeddings)  # Directly use embeddings if they're numpy arrays
    clustering_model = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=2000)
    clustering_model.fit(embeddings_array)
    print(f"Clustering completed. Labels: {np.unique(clustering_model.labels_)}")
    return clustering_model.labels_

def process_audio(file_path: str, size_of_model: str, task: str, source_language: Optional[str], speaker_number: Optional[int]) -> dict:
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
            "language": source_language if source_language else None
        }
    )

    waveform, sample_rate = load_audio_file(converted_audio_file_path)
    
    # Start timing for transcription
    start_transcription_time = time.time()
    transcription_result = perform_transcription(asr_pipeline, converted_audio_file_path)
    # Calculate and print transcription metrics
    end_transcription_time = time.time()
    transcription_time = end_transcription_time - start_transcription_time
    audio_length_seconds = torchaudio.info(converted_audio_file_path).num_frames / sample_rate
    audio_length_minutes = audio_length_seconds / 60
    transcription_speed = audio_length_minutes / (transcription_time / 60)
    print(f"Total minutes transcribed: {audio_length_minutes:.2f} minutes")
    print(f"Transcription speed: {transcription_speed:.2f} minutes transcribed per minute")
    
    del asr_pipeline
    
    segments = [(chunk["timestamp"][0], chunk["timestamp"][1]) for chunk in transcription_result["chunks"]]

    if speaker_number == 0:
        embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device="cuda")
    else:
        embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device="cuda")
    
    # Start timing for embeddings
    start_embedding_time = time.time()
    embeddings = generate_embeddings_for_segments(embedding_model, waveform, sample_rate, segments)
    # Calculate and print embedding metrics
    end_embedding_time = time.time()
    embedding_time = end_embedding_time - start_embedding_time
    segments_per_minute = len(segments) / (embedding_time / 60)
    print(f"Segments embedded per minute: {segments_per_minute:.2f}")
    
    if speaker_number == 0:
        speaker_labels = cluster_embeddings(embeddings)
    else:
        clustering_model = AgglomerativeClustering(n_clusters=speaker_number)
        speaker_labels = clustering_model.fit_predict(np.vstack(embeddings))

    grouped_segments = generate_transcription_output(transcription_result["chunks"], speaker_labels)
    
    del embedding_model
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    os.remove(file_path)
    os.remove(converted_audio_file_path)
    
    return grouped_segments


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...), 
                           size_of_model: str = Form(...), 
                           task: str = Form(...), 
                           source_language: Optional[str] = Form(None), 
                           speaker_number: Optional[int] = Form(0)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    # Wrap the processing in a separate process
    with multiprocessing.Pool(1) as pool:
        async_result = pool.apply_async(process_audio, (temp_file_path, size_of_model, task, source_language, speaker_number))
        result = async_result.get()  # Wait for the process to complete and get the result

    return JSONResponse(content=result)

# Remember to run the FastAPI app using a command like: uvicord whisper-api:app --reload
