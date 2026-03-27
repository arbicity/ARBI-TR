"""
Integration tests — require a running ARBI-TR server.
See conftest.py for setup instructions.

Uses bundled tests/fixtures/agi_clip.ogg (two-speaker real speech, ~1.6 MB).
Override with TEST_AUDIO_FILE=/path/to/file for custom audio.
"""

import time

import httpx
import pytest

from tests.integration.conftest import wait_for_task


def test_health(base_url):
    resp = httpx.get(f"{base_url}/health", timeout=5)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["queue_length"], int)


def test_task_status_unknown_session(base_url):
    resp = httpx.get(f"{base_url}/task_status/does-not-exist", timeout=5)
    assert resp.status_code == 404


def test_transcribe_with_diarization(base_url, audio_wav):
    """Full pipeline: transcribe + diarize two-speaker audio, check schema and speakers."""
    resp = httpx.post(
        f"{base_url}/transcribe/",
        data={"size_of_model": "large", "task_str": "transcribe", "speaker_number": "2"},
        files={"file": ("test.ogg", audio_wav, "audio/ogg")},
        timeout=30,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert body["message"] == "Your request is queued for processing"

    result = wait_for_task(base_url, body["session_id"])
    assert result["status"] == "completed"
    assert len(result["segments"]) > 0

    # Schema
    for seg in result["segments"]:
        assert set(seg.keys()) == {"Start", "End", "Speaker", "Text"}
        assert seg["Speaker"].startswith("SPEAKER_")

    # Two-speaker audio should produce 2 speakers
    speakers = {s["Speaker"] for s in result["segments"]}
    assert len(speakers) == 2, f"Expected 2 speakers, got {speakers}"

    # Non-trivial text
    total_text = " ".join(s["Text"] for s in result["segments"])
    assert len(total_text.strip()) > 100


def test_openai_transcription(base_url, audio_wav):
    resp = httpx.post(
        f"{base_url}/v1/audio/transcriptions",
        data={"model": "whisper-large-v3", "language": "en"},
        files={"file": ("test.ogg", audio_wav, "audio/ogg")},
        timeout=120,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["text"], str)
    assert len(body["text"]) > 100


def test_openai_transcription_missing_file(base_url):
    resp = httpx.post(
        f"{base_url}/v1/audio/transcriptions",
        data={"model": "whisper-large-v3"},
        timeout=10,
    )
    assert resp.status_code == 422


def test_openai_translation(base_url, audio_wav):
    resp = httpx.post(
        f"{base_url}/v1/audio/translations",
        data={"model": "whisper-large-v3"},
        files={"file": ("test.ogg", audio_wav, "audio/ogg")},
        timeout=120,
    )
    assert resp.status_code == 200
    assert isinstance(resp.json()["text"], str)
