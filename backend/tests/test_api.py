"""
API endpoint tests — models are mocked so no GPU/HF_TOKEN required.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Patch model loading before importing app so lifespan doesn't try to load real models
@pytest.fixture(autouse=True)
def mock_models():
    with (
        patch("utils.get_whisper_model", return_value=MagicMock()),
        patch("utils.get_batched_pipeline", return_value=MagicMock()),
        patch("utils.get_diarization_pipeline", return_value=MagicMock()),
        patch("utils.initialize_models", return_value=None),
    ):
        yield


@pytest.fixture()
def client():
    from main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


FAKE_AUDIO = b"RIFF\x00\x00\x00\x00WAVEfmt "  # minimal fake WAV header

MOCK_SEGMENTS = {
    "segments": [
        {"Start": "0:00:01", "End": "0:00:05", "Speaker": "SPEAKER_00", "Text": "Hello world"}
    ]
}

MOCK_TEXT = {"text": "Hello world"}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "queue_length" in body


# ---------------------------------------------------------------------------
# /transcribe/ — async queue
# ---------------------------------------------------------------------------


def test_transcribe_returns_session_id(client):
    with patch("main.process_audio", return_value=MOCK_SEGMENTS):
        resp = client.post(
            "/transcribe/",
            data={"size_of_model": "large", "task_str": "transcribe"},
            files={"file": ("test.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert body["message"] == "Your request is queued for processing"
    assert "queue_position" in body


def test_transcribe_missing_file(client):
    resp = client.post(
        "/transcribe/",
        data={"size_of_model": "large", "task_str": "transcribe"},
    )
    assert resp.status_code == 422


def test_transcribe_missing_task_str(client):
    resp = client.post(
        "/transcribe/",
        data={"size_of_model": "large"},
        files={"file": ("test.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /task_status/{session_id}
# ---------------------------------------------------------------------------


def test_task_status_not_found(client):
    resp = client.get("/task_status/nonexistent-session-id")
    assert resp.status_code == 404


def test_task_status_queued(client):
    """Submit a job then immediately check status (it will be queued or completed)."""
    with patch("main.process_audio", return_value=MOCK_SEGMENTS):
        post = client.post(
            "/transcribe/",
            data={"size_of_model": "large", "task_str": "transcribe"},
            files={"file": ("test.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )
    session_id = post.json()["session_id"]
    resp = client.get(f"/task_status/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("queued", "completed", "failed")


def test_task_status_completed_has_segments(client):
    """Force a task into completed state and verify segments are returned."""
    import main as m

    session_id = "test-completed-id"
    m.tasks[session_id] = {"status": "completed", "data": MOCK_SEGMENTS}

    resp = client.get(f"/task_status/{session_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert len(body["segments"]) == 1
    assert body["segments"][0]["Speaker"] == "SPEAKER_00"


def test_task_status_failed_has_error(client):
    import main as m

    session_id = "test-failed-id"
    m.tasks[session_id] = {"status": "failed", "error": "CUDA out of memory"}

    resp = client.get(f"/task_status/{session_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert "CUDA out of memory" in body["error"]


# ---------------------------------------------------------------------------
# /v1/audio/transcriptions (OpenAI-compatible)
# ---------------------------------------------------------------------------


def test_openai_transcription(client):
    with patch(
        "main.process_audio_without_diarization", return_value=MOCK_TEXT
    ):
        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-large-v3"},
            files={"file": ("test.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )
    assert resp.status_code == 200
    assert resp.json()["text"] == "Hello world"


def test_openai_transcription_with_language(client):
    with patch(
        "main.process_audio_without_diarization", return_value=MOCK_TEXT
    ) as mock_fn:
        client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-large-v3", "language": "fr"},
            files={"file": ("test.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )
        # Verify language was passed through
        _, kwargs = mock_fn.call_args
        assert kwargs.get("source_language") == "fr" or mock_fn.call_args[0][2] == "fr"


def test_openai_transcription_missing_file(client):
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-large-v3"},
    )
    assert resp.status_code == 422


def test_openai_transcription_error_propagates(client):
    with patch(
        "main.process_audio_without_diarization",
        side_effect=RuntimeError("FFmpeg not found"),
    ):
        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-large-v3"},
            files={"file": ("test.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )
    assert resp.status_code == 500
    assert "FFmpeg not found" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /v1/audio/translations (OpenAI-compatible)
# ---------------------------------------------------------------------------


def test_openai_translation(client):
    with patch(
        "main.process_audio_without_diarization", return_value=MOCK_TEXT
    ) as mock_fn:
        resp = client.post(
            "/v1/audio/translations",
            data={"model": "whisper-large-v3"},
            files={"file": ("test.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )
    assert resp.status_code == 200
    assert resp.json()["text"] == "Hello world"
    # Translation always passes task="translate" and language=None
    call_args = mock_fn.call_args[0]
    assert call_args[1] == "translate"
    assert call_args[2] is None
