"""
Integration tests — require a running ARBI-TR server.
See conftest.py for setup instructions.

Set TEST_AUDIO_FILE=/path/to/real.wav for meaningful transcription assertions.
Without it, a synthetic WAV is used (pipeline correctness only, not content).
"""

import os
import time

import httpx
import pytest

from tests.integration.conftest import wait_for_task

HAS_REAL_AUDIO = bool(os.getenv("TEST_AUDIO_FILE"))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health(base_url):
    resp = httpx.get(f"{base_url}/health", timeout=5)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["queue_length"], int)


# ---------------------------------------------------------------------------
# /transcribe/ — async queue with diarization
# ---------------------------------------------------------------------------


def test_transcribe_queues_and_returns_session_id(base_url, synthetic_wav):
    resp = httpx.post(
        f"{base_url}/transcribe/",
        data={"size_of_model": "large", "task_str": "transcribe"},
        files={"file": ("test.wav", synthetic_wav, "audio/wav")},
        timeout=30,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert "queue_position" in body
    assert body["message"] == "Your request is queued for processing"


def test_transcribe_completes(base_url, synthetic_wav):
    resp = httpx.post(
        f"{base_url}/transcribe/",
        data={"size_of_model": "large", "task_str": "transcribe"},
        files={"file": ("test.wav", synthetic_wav, "audio/wav")},
        timeout=30,
    )
    session_id = resp.json()["session_id"]
    result = wait_for_task(base_url, session_id)

    assert result["status"] == "completed"
    assert "segments" in result
    assert isinstance(result["segments"], list)


def test_transcribe_segment_schema(base_url, synthetic_wav):
    """Each returned segment must have all required fields with correct types."""
    resp = httpx.post(
        f"{base_url}/transcribe/",
        data={"size_of_model": "large", "task_str": "transcribe"},
        files={"file": ("test.wav", synthetic_wav, "audio/wav")},
        timeout=30,
    )
    session_id = resp.json()["session_id"]
    result = wait_for_task(base_url, session_id)

    for seg in result["segments"]:
        assert set(seg.keys()) == {"Start", "End", "Speaker", "Text"}, (
            f"Unexpected segment keys: {seg.keys()}"
        )
        assert isinstance(seg["Start"], str)
        assert isinstance(seg["End"], str)
        assert isinstance(seg["Speaker"], str)
        assert isinstance(seg["Text"], str)
        assert seg["Speaker"].startswith("SPEAKER_")


def test_transcribe_speaker_number_respected(base_url, synthetic_wav):
    """When speaker_number=1 is set, there should be exactly 1 unique speaker."""
    resp = httpx.post(
        f"{base_url}/transcribe/",
        data={"size_of_model": "large", "task_str": "transcribe", "speaker_number": "1"},
        files={"file": ("test.wav", synthetic_wav, "audio/wav")},
        timeout=30,
    )
    session_id = resp.json()["session_id"]
    result = wait_for_task(base_url, session_id)

    if result["segments"]:
        speakers = {s["Speaker"] for s in result["segments"]}
        assert len(speakers) == 1, f"Expected 1 speaker, got {speakers}"


@pytest.mark.skipif(not HAS_REAL_AUDIO, reason="TEST_AUDIO_FILE not set")
def test_transcribe_real_audio_non_empty(base_url, synthetic_wav):
    """With real speech audio, transcription must produce non-empty segments."""
    resp = httpx.post(
        f"{base_url}/transcribe/",
        data={"size_of_model": "large", "task_str": "transcribe"},
        files={"file": ("real.wav", synthetic_wav, "audio/wav")},
        timeout=30,
    )
    session_id = resp.json()["session_id"]
    result = wait_for_task(base_url, session_id)

    assert len(result["segments"]) > 0, "Real audio produced zero segments"
    total_text = " ".join(s["Text"] for s in result["segments"])
    assert len(total_text.strip()) > 10, "Transcribed text is too short"


# ---------------------------------------------------------------------------
# /task_status/{session_id}
# ---------------------------------------------------------------------------


def test_task_status_unknown_session(base_url):
    resp = httpx.get(f"{base_url}/task_status/does-not-exist", timeout=5)
    assert resp.status_code == 404


def test_task_status_transitions(base_url, synthetic_wav):
    """Status must go from queued → completed (never backwards)."""
    resp = httpx.post(
        f"{base_url}/transcribe/",
        data={"size_of_model": "large", "task_str": "transcribe"},
        files={"file": ("test.wav", synthetic_wav, "audio/wav")},
        timeout=30,
    )
    session_id = resp.json()["session_id"]

    seen_statuses = []
    deadline = time.time() + 300
    while time.time() < deadline:
        s = httpx.get(f"{base_url}/task_status/{session_id}", timeout=5).json()
        status = s["status"]
        if not seen_statuses or seen_statuses[-1] != status:
            seen_statuses.append(status)
        if status in ("completed", "failed"):
            break
        time.sleep(2)

    valid_sequence = {"queued", "completed"} | {"failed"}
    assert set(seen_statuses) <= valid_sequence
    assert seen_statuses[-1] == "completed", f"Final status: {seen_statuses}"


# ---------------------------------------------------------------------------
# /v1/audio/transcriptions (OpenAI-compatible)
# ---------------------------------------------------------------------------


def test_openai_transcription_returns_text(base_url, synthetic_wav):
    resp = httpx.post(
        f"{base_url}/v1/audio/transcriptions",
        data={"model": "whisper-large-v3"},
        files={"file": ("test.wav", synthetic_wav, "audio/wav")},
        timeout=120,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "text" in body
    assert isinstance(body["text"], str)


def test_openai_transcription_with_language(base_url, synthetic_wav):
    resp = httpx.post(
        f"{base_url}/v1/audio/transcriptions",
        data={"model": "whisper-large-v3", "language": "en"},
        files={"file": ("test.wav", synthetic_wav, "audio/wav")},
        timeout=120,
    )
    assert resp.status_code == 200
    assert "text" in resp.json()


def test_openai_transcription_missing_file(base_url):
    resp = httpx.post(
        f"{base_url}/v1/audio/transcriptions",
        data={"model": "whisper-large-v3"},
        timeout=10,
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /v1/audio/translations (OpenAI-compatible)
# ---------------------------------------------------------------------------


def test_openai_translation_returns_text(base_url, synthetic_wav):
    resp = httpx.post(
        f"{base_url}/v1/audio/translations",
        data={"model": "whisper-large-v3"},
        files={"file": ("test.wav", synthetic_wav, "audio/wav")},
        timeout=120,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "text" in body
    assert isinstance(body["text"], str)


# ---------------------------------------------------------------------------
# Concurrent requests (basic queue behaviour)
# ---------------------------------------------------------------------------


def test_multiple_requests_queue_correctly(base_url, synthetic_wav):
    """Submit 3 jobs concurrently and verify all complete successfully."""
    session_ids = []
    for _ in range(3):
        resp = httpx.post(
            f"{base_url}/transcribe/",
            data={"size_of_model": "large", "task_str": "transcribe"},
            files={"file": ("test.wav", synthetic_wav, "audio/wav")},
            timeout=30,
        )
        assert resp.status_code == 200
        session_ids.append(resp.json()["session_id"])

    for sid in session_ids:
        result = wait_for_task(base_url, sid, timeout_s=600)
        assert result["status"] == "completed"
        assert "segments" in result
