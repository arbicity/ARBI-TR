"""
ARBI-TR processing utilities.

Pipeline:
  1. FFmpeg converts input to 16 kHz mono WAV
  2. faster-whisper transcribes (large-v3, CUDA float16)
  3. pyannote speaker-diarization-community-1 diarizes (pyannote 4.x)
  4. Segments merged by max-overlap speaker assignment + consecutive-speaker merging
"""

import datetime
import gc
import os
import subprocess
import tempfile
import time
from typing import Optional

import numpy as np
import torch
from loguru import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
WHISPER_BEAM_SIZE: int = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
PYANNOTE_MODEL: str = os.getenv(
    "PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"
)
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

# ---------------------------------------------------------------------------
# Model singletons — loaded once, reused across requests
# ---------------------------------------------------------------------------

_whisper_model = None
_diarization_pipeline = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_whisper_model():
    """Return the faster-whisper WhisperModel singleton, loading it if needed."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        device = _device()
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info(
            f"Loading faster-whisper {WHISPER_MODEL_SIZE} on {device} ({compute_type})"
        )
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=device,
            compute_type=compute_type,
        )
        logger.info("faster-whisper model loaded")
    return _whisper_model


def get_diarization_pipeline():
    """Return the pyannote diarization Pipeline singleton, loading it if needed.

    HF_TOKEN is only required for the initial model download. In production
    models are pre-cached at HF_HOME (/mnt/k8scache) and HF_HUB_OFFLINE=1
    so no token is needed at runtime.
    """
    global _diarization_pipeline
    if _diarization_pipeline is None:
        from pyannote.audio import Pipeline

        logger.info(f"Loading pyannote pipeline: {PYANNOTE_MODEL}")
        _diarization_pipeline = Pipeline.from_pretrained(
            PYANNOTE_MODEL,
            token=HF_TOKEN or None,  # None = use cached, token = download
        )
        _diarization_pipeline.to(torch.device(_device()))
        logger.info("pyannote diarization pipeline loaded")
    return _diarization_pipeline


def initialize_models() -> None:
    """Eagerly load all models at startup to avoid first-request latency."""
    get_whisper_model()
    get_diarization_pipeline()


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------


def convert_audio_to_wav(media_file_path: str) -> str:
    """Convert any audio/video file to 16 kHz mono PCM WAV via FFmpeg."""
    t0 = time.perf_counter()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    wav_path = tmp.name
    try:
        subprocess.run(  # nosec B603 B607
            [
                "ffmpeg", "-y", "-i", media_file_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                wav_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        raise
    logger.debug(f"convert_audio_to_wav: {time.perf_counter() - t0:.2f}s")
    return wav_path


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe_audio(
    wav_path: str,
    task: str = "transcribe",
    language: Optional[str] = None,
) -> list[dict]:
    """
    Transcribe audio with faster-whisper.

    Returns a list of chunk dicts: {"timestamp": (start, end), "text": str}
    """
    t0 = time.perf_counter()
    model = get_whisper_model()
    segments_gen, info = model.transcribe(
        wav_path,
        task=task,
        language=language or None,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    # Consume the generator — faster-whisper is lazy
    chunks = [
        {"timestamp": (seg.start, seg.end), "text": seg.text}
        for seg in segments_gen
    ]
    logger.info(
        f"Transcribed {len(chunks)} segments in {time.perf_counter() - t0:.2f}s "
        f"(detected language: {info.language}, prob: {info.language_probability:.2f})"
    )
    return chunks


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------


def diarize_audio(
    wav_path: str,
    num_speakers: Optional[int] = None,
) -> list[dict]:
    """
    Run pyannote speaker diarization.

    Returns list of {"start": float, "end": float, "speaker": str}.
    If num_speakers is 0 or None the pipeline auto-detects the count.
    """
    t0 = time.perf_counter()
    pipeline = get_diarization_pipeline()

    kwargs: dict = {}
    if num_speakers and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers

    # Preload audio as tensor dict — avoids torchcodec/AudioDecoder dependency
    import soundfile as sf
    waveform, sample_rate = sf.read(wav_path, dtype="float32")
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    else:
        waveform = waveform.T  # [samples, channels] -> [channels, samples]
    audio_input = {"waveform": torch.from_numpy(waveform), "sample_rate": sample_rate}

    output = pipeline(audio_input, **kwargs)
    # pyannote 4.x returns DiarizeOutput; use exclusive turns (no overlap) for transcription alignment
    annotation = output.exclusive_speaker_diarization if hasattr(output, "exclusive_speaker_diarization") else output
    segments = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    logger.info(
        f"Diarized {len(segments)} speaker segments in {time.perf_counter() - t0:.2f}s"
    )
    return segments


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def convert_time(secs: float) -> str:
    """Format seconds as HH:MM:SS."""
    if secs is None:
        return "00:00:00"
    return str(datetime.timedelta(seconds=round(secs)))


def _assign_speaker(
    chunk_start: float,
    chunk_end: float,
    diarization_segments: list[dict],
) -> str:
    """Return the speaker with maximum overlap with [chunk_start, chunk_end]."""
    best_speaker = "SPEAKER_00"
    max_overlap = 0.0
    for seg in diarization_segments:
        overlap = max(
            0.0,
            min(chunk_end, seg["end"]) - max(chunk_start, seg["start"]),
        )
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = seg["speaker"]
    return best_speaker


def merge_transcription_with_diarization(
    chunks: list[dict],
    diarization_segments: list[dict],
    merge_gap_s: float = 0.5,
) -> dict:
    """
    Assign a speaker to each Whisper chunk by max-overlap with pyannote output,
    then merge consecutive chunks from the same speaker.

    merge_gap_s: maximum silence gap (seconds) between consecutive same-speaker
                 chunks that should still be merged into one segment.
    """
    if not chunks:
        return {"segments": []}

    labelled: list[dict] = []
    for chunk in chunks:
        ts = chunk.get("timestamp")
        if not ts or ts[1] is None:
            continue
        text = chunk.get("text", "").strip()
        if not text:
            continue
        speaker = _assign_speaker(ts[0], ts[1], diarization_segments)
        labelled.append(
            {
                "start": ts[0],
                "end": ts[1],
                "speaker": speaker,
                "text": text,
            }
        )

    if not labelled:
        return {"segments": []}

    # Merge consecutive segments from the same speaker within merge_gap_s
    merged: list[dict] = [labelled[0].copy()]
    for seg in labelled[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if seg["speaker"] == prev["speaker"] and gap <= merge_gap_s:
            prev["end"] = seg["end"]
            prev["text"] = prev["text"] + " " + seg["text"]
        else:
            merged.append(seg.copy())

    return {
        "segments": [
            {
                "Start": convert_time(s["start"]),
                "End": convert_time(s["end"]),
                "Speaker": s["speaker"],
                "Text": s["text"],
            }
            for s in merged
        ]
    }


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def process_audio(
    file_path: str,
    task: str,
    source_language: Optional[str],
    num_speakers: Optional[int],
) -> dict:
    """Transcribe + diarize an audio/video file. Returns {segments: [...]}."""
    wav_path: Optional[str] = None
    try:
        wav_path = convert_audio_to_wav(file_path)
        chunks = transcribe_audio(wav_path, task=task, language=source_language)
        if not chunks:
            return {"segments": []}
        diar_segs = diarize_audio(wav_path, num_speakers=num_speakers)
        return merge_transcription_with_diarization(chunks, diar_segs)
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


def process_audio_without_diarization(
    file_path: str,
    task: str,
    source_language: Optional[str],
) -> dict:
    """Transcribe only (no diarization). Returns {text: str}."""
    wav_path: Optional[str] = None
    try:
        wav_path = convert_audio_to_wav(file_path)
        chunks = transcribe_audio(wav_path, task=task, language=source_language)
        text = " ".join(c["text"].strip() for c in chunks)
        return {"text": text}
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
