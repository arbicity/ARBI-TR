"""
Integration test fixtures.

Requires a running ARBI-TR server. Start it with:
    ./scripts/run-integration-tests.sh

Or manually:
    HF_TOKEN=hf_xxx docker compose -f docker-compose.test.yaml up --build -d
    # wait for /health to return 200
    uv run pytest tests/integration/ -v
    docker compose -f docker-compose.test.yaml down

Uses tests/fixtures/agi_clip.ogg — a 10-min two-speaker clip
(Lex Fridman + Jensen Huang discussing AGI) compressed to ~1.6 MB.
Override with TEST_AUDIO_FILE env var.
"""

import os
import time
from pathlib import Path

import httpx
import pytest

BASE_URL = os.getenv("ARBI_TR_URL", "http://localhost:8000")
POLL_INTERVAL_S = 2
TASK_TIMEOUT_S = 300  # 5 min — model load + transcription on cold start

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
DEFAULT_AUDIO = FIXTURES_DIR / "agi_clip.ogg"


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session", autouse=True)
def require_server():
    """Fail fast if server is not reachable."""
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    pytest.fail(
        f"ARBI-TR server not reachable at {BASE_URL}. "
        "Start it with: ./scripts/run-integration-tests.sh"
    )


@pytest.fixture(scope="session")
def audio_wav() -> bytes:
    """
    Return audio bytes for integration tests.

    Uses tests/fixtures/agi_clip.ogg — a 10-min two-speaker clip
    (Lex Fridman + Jensen Huang) at 16 kHz mono opus ~1.6 MB.
    Override with TEST_AUDIO_FILE env var.
    """
    test_audio = os.getenv("TEST_AUDIO_FILE")
    if test_audio and Path(test_audio).exists():
        return Path(test_audio).read_bytes()

    if DEFAULT_AUDIO.exists():
        return DEFAULT_AUDIO.read_bytes()

    pytest.fail(
        f"{DEFAULT_AUDIO} not found. "
        "Set TEST_AUDIO_FILE or restore the fixture."
    )


def wait_for_task(base_url: str, session_id: str, timeout_s: int = TASK_TIMEOUT_S) -> dict:
    """Poll /task_status/{session_id} until done or timeout. Raises on timeout/failure."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = httpx.get(f"{base_url}/task_status/{session_id}", timeout=10)
        resp.raise_for_status()
        body = resp.json()
        status = body["status"]
        if status == "completed":
            return body
        if status == "failed":
            pytest.fail(f"Task {session_id} failed: {body.get('error')}")
        time.sleep(POLL_INTERVAL_S)
    pytest.fail(f"Task {session_id} did not complete within {timeout_s}s")
