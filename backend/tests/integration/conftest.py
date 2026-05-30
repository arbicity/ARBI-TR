"""
Integration test fixtures — exercise the service through the AUTO-GENERATED client
(`arbi_tr_client`), so the published client and the server are validated together.

Requires a running ARBI-TR server. Start it with:
    ./scripts/run-integration-tests.sh

Or manually:
    HF_TOKEN=hf_xxx docker compose -f docker-compose.test.yaml up --build -d
    bash scripts/generate-client.sh          # builds + installs arbi_tr_client
    ARBI_TR_URL=http://localhost:8000 uv run --project backend pytest backend/tests/integration -v
    docker compose -f docker-compose.test.yaml down

Uses tests/fixtures/agi_clip.ogg — a 10-min two-speaker clip
(Lex Fridman + Jensen Huang discussing AGI) compressed to ~1.6 MB.
Override with TEST_AUDIO_FILE env var.
"""

import os
import time
from pathlib import Path

import pytest
from arbi_tr_client import Client
from arbi_tr_client.api.default import (
    get_task_status_task_status_session_id_get as _task_status,
)
from arbi_tr_client.models import TaskStatusResponse

BASE_URL = os.getenv("ARBI_TR_URL", "http://localhost:8000")
POLL_INTERVAL_S = 2
TASK_TIMEOUT_S = 300  # 5 min — model load + transcription on cold start

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
DEFAULT_AUDIO = FIXTURES_DIR / "agi_clip.ogg"


@pytest.fixture(scope="session")
def client() -> Client:
    """The auto-generated ARBI-TR API client, pointed at the test server."""
    return Client(base_url=BASE_URL, timeout=120.0, raise_on_unexpected_status=False)


@pytest.fixture(scope="session", autouse=True)
def require_server(client: Client):
    """Fail fast if the server is not reachable (probe /health through the client)."""
    from arbi_tr_client.api.default import health_health_get

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if health_health_get.sync_detailed(client=client).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    pytest.fail(f"ARBI-TR server not reachable at {BASE_URL}. " "Start it with: ./scripts/run-integration-tests.sh")


@pytest.fixture(scope="session")
def audio_bytes() -> bytes:
    """Audio bytes for integration tests (tests/fixtures/agi_clip.ogg, override with TEST_AUDIO_FILE)."""
    test_audio = os.getenv("TEST_AUDIO_FILE")
    if test_audio and Path(test_audio).exists():
        return Path(test_audio).read_bytes()
    if DEFAULT_AUDIO.exists():
        return DEFAULT_AUDIO.read_bytes()
    pytest.fail(f"{DEFAULT_AUDIO} not found. Set TEST_AUDIO_FILE or restore the fixture.")


def wait_for_task(client: Client, session_id: str, timeout_s: int = TASK_TIMEOUT_S) -> TaskStatusResponse:
    """Poll /task_status/{session_id} via the client until done or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = _task_status.sync(client=client, session_id=session_id)
        if isinstance(resp, TaskStatusResponse):
            if resp.status == "completed":
                return resp
            if resp.status == "failed":
                pytest.fail(f"Task {session_id} failed: {resp.error}")
        time.sleep(POLL_INTERVAL_S)
    pytest.fail(f"Task {session_id} did not complete within {timeout_s}s")
