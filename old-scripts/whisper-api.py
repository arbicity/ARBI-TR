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

app = FastAPI()

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
    """Converts an audio file to WAV format using ffmpeg and returns the path of the temporary WAV file."""
    # Create a temporary file for the converted WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        audio_file_path = temp_wav_file.name
    
    # Perform the conversion
    try:
        cmd = ['ffmpeg', '-y', '-i', media_file_path, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', audio_file_path]
        subprocess.run(cmd, check=True)
        print(f"Converted {media_file_path} to WAV: {audio_file_path}")
    except Exception as e:
        # If conversion fails, delete the temporary file and re-raise the exception
        os.remove(audio_file_path)
        raise e

    return audio_file_path

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

def cluster_embeddings(embeddings: List[torch.Tensor]) -> np.ndarray:
    """Clusters embeddings and returns the labels."""
    embeddings_array = np.vstack(embeddings)  # Directly use embeddings if they're numpy arrays
    clustering_model = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=2000)
    clustering_model.fit(embeddings_array)
    print(f"Clustering completed. Labels: {np.unique(clustering_model.labels_)}")
    return clustering_model.labels_

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...), 
                           size_of_model: str = Form(...), 
                           task: str = Form(...), 
                           source_language: Optional[str] = Form(None), 
                           speaker_number: Optional[int] = Form(0)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    converted_audio_file_path = convert_audio_to_wav(temp_file_path)

    # Select the correct model based on user input
    model_id = f"openai/whisper-{size_of_model}" + ("-v3" if size_of_model == "large" else "")

    # Initialize the ASR pipeline with dynamic parameters
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
    
    transcription_result = perform_transcription(asr_pipeline, converted_audio_file_path)
    segments = [(chunk["timestamp"][0], chunk["timestamp"][1]) for chunk in transcription_result["chunks"]]
    
    del asr_pipeline
    # Speaker embedding and clustering
    if speaker_number == 0:
        # Automatic speaker number detection and clustering
        embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device="cuda")  # or "cuda"
        embeddings = generate_embeddings_for_segments(embedding_model, waveform, sample_rate, segments)
        speaker_labels = cluster_embeddings(embeddings)
    else:
        # Process with known speaker number but still distinguish each speaker
        embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device="cuda")  # or "cuda"
        embeddings = generate_embeddings_for_segments(embedding_model, waveform, sample_rate, segments)
        # Apply a simplified clustering or labeling process since the number of speakers is known
        clustering_model = AgglomerativeClustering(n_clusters=speaker_number)
        speaker_labels = clustering_model.fit_predict(np.vstack(embeddings))

    grouped_segments = generate_transcription_output(transcription_result["chunks"], speaker_labels)
    del embedding_model
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
    os.remove(temp_file_path)
    os.remove(converted_audio_file_path)

    return JSONResponse(content=grouped_segments)

# Remember to run the FastAPI app using a command like: uvicord whisper-api:app --reload
