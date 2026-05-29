"""
Integration tests — drive a running ARBI-TR server entirely through the
auto-generated `arbi_tr_client`, validating the client and the API together.
See conftest.py for setup instructions.

Uses bundled tests/fixtures/agi_clip.ogg (two-speaker real speech, ~1.6 MB).
Override with TEST_AUDIO_FILE=/path/to/file for custom audio.
"""

import io

import httpx
import pytest
from arbi_tr_client import Client
from arbi_tr_client.api.default import (
    get_task_status_task_status_session_id_get as task_status,
)
from arbi_tr_client.api.default import health_health_get
from arbi_tr_client.api.default import (
    transcribe_audio_endpoint_transcribe_post as transcribe,
)
from arbi_tr_client.api.default import (
    transcribe_openai_v1_audio_transcriptions_post as openai_transcribe,
)
from arbi_tr_client.api.default import (
    translate_openai_v1_audio_translations_post as openai_translate,
)
from arbi_tr_client.models import (
    BodyTranscribeAudioEndpointTranscribePost,
    BodyTranscribeOpenaiV1AudioTranscriptionsPost,
    BodyTranslateOpenaiV1AudioTranslationsPost,
    HealthResponse,
    TaskStatusResponse,
    TranscribeSubmitResponse,
    Transcription,
)
from arbi_tr_client.types import File
from tests.integration.conftest import BASE_URL, wait_for_task

pytestmark = pytest.mark.integration


def _audio_file(audio_bytes: bytes) -> File:
    return File(payload=io.BytesIO(audio_bytes), file_name="test.ogg", mime_type="audio/ogg")


def test_health(client: Client):
    result = health_health_get.sync(client=client)
    assert isinstance(result, HealthResponse)
    assert result.status == "ok"
    assert isinstance(result.queue_length, int)


def test_task_status_unknown_session(client: Client):
    resp = task_status.sync_detailed(client=client, session_id="does-not-exist")
    assert resp.status_code == 404


def test_transcribe_with_diarization(client: Client, audio_bytes: bytes):
    """Full pipeline: transcribe + diarize two-speaker audio; check typed schema and speakers."""
    body = BodyTranscribeAudioEndpointTranscribePost(
        file=_audio_file(audio_bytes),
        size_of_model="large",
        task_str="transcribe",
        speaker_number=2,
    )
    submitted = transcribe.sync(client=client, body=body)
    assert isinstance(submitted, TranscribeSubmitResponse)
    assert submitted.session_id
    assert submitted.message == "Your request is queued for processing"

    result = wait_for_task(client, submitted.session_id)
    assert isinstance(result, TaskStatusResponse)
    assert result.status == "completed"
    assert result.segments and len(result.segments) > 0

    # Typed segment schema (Segment model: Start/End/Speaker/Text).
    for seg in result.segments:
        assert seg.speaker.startswith("SPEAKER_")
        assert seg.start and seg.end

    # Two-speaker audio should yield 2 distinct speakers.
    speakers = {seg.speaker for seg in result.segments}
    assert len(speakers) == 2, f"Expected 2 speakers, got {speakers}"

    # Non-trivial transcript.
    total_text = " ".join(seg.text for seg in result.segments)
    assert len(total_text.strip()) > 100


def test_openai_transcription(client: Client, audio_bytes: bytes):
    body = BodyTranscribeOpenaiV1AudioTranscriptionsPost(
        file=_audio_file(audio_bytes),
        model="whisper-large-v3",
        language="en",
    )
    result = openai_transcribe.sync(client=client, body=body)
    assert isinstance(result, Transcription)
    assert len(result.text) > 100


def test_openai_translation(client: Client, audio_bytes: bytes):
    body = BodyTranslateOpenaiV1AudioTranslationsPost(
        file=_audio_file(audio_bytes),
        model="whisper-large-v3",
    )
    result = openai_translate.sync(client=client, body=body)
    assert isinstance(result, Transcription)
    assert isinstance(result.text, str)


def test_openai_transcription_missing_file():
    """Negative case: the typed client requires `file`, so post a fileless multipart
    request directly to assert the server still returns 422."""
    resp = httpx.post(
        f"{BASE_URL}/v1/audio/transcriptions",
        data={"model": "whisper-large-v3"},
        timeout=10,
    )
    assert resp.status_code == 422
