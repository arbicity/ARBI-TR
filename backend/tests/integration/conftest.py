"""
Integration test fixtures.

Requires a running ARBI-TR server. Start it with:
    ./scripts/run-integration-tests.sh

Or manually:
    HF_TOKEN=hf_xxx docker compose -f docker-compose.test.yaml up --build -d
    # wait for /health to return 200
    uv run pytest tests/integration/ -v
    docker compose -f docker-compose.test.yaml down
"""

import math
import os
import struct
import time
import wave
from pathlib import Path

import httpx
import pytest

BASE_URL = os.getenv("ARBI_TR_URL", "http://localhost:8000")
POLL_INTERVAL_S = 2
TASK_TIMEOUT_S = 300  # 5 min — model load + transcription on cold start


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
def synthetic_wav(tmp_path_factory) -> bytes:
    """
    Generate a short synthetic WAV: 4 seconds of speech-like noise
    (white noise modulated at human voice cadence frequencies).
    Whisper/VAD may or may not find speech — we test pipeline correctness,
    not transcription accuracy. Replace with TEST_AUDIO_FILE for real accuracy tests.
    """
    # If user provides a real audio file, use it instead
    test_audio = os.getenv("TEST_AUDIO_FILE")
    if test_audio and Path(test_audio).exists():
        return Path(test_audio).read_bytes()

    sample_rate = 16000
    duration_s = 4
    num_samples = sample_rate * duration_s
    tmp = tmp_path_factory.mktemp("audio") / "test.wav"

    # White noise with amplitude modulated by a 3 Hz envelope (voice cadence)
    # and bandpass-shaped toward the 300–3000 Hz speech range
    import random
    rng = random.Random(42)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 3 * t)
        noise = rng.uniform(-1, 1)
        # Add formant-like tones at 500, 1500, 2500 Hz
        formants = (
            math.sin(2 * math.pi * 500 * t) * 0.3
            + math.sin(2 * math.pi * 1500 * t) * 0.2
            + math.sin(2 * math.pi * 2500 * t) * 0.1
        )
        sample = (noise * 0.4 + formants) * envelope
        samples.append(max(-1.0, min(1.0, sample)))

    with wave.open(str(tmp), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        data = struct.pack(f"<{num_samples}h", *[int(s * 32767) for s in samples])
        wf.writeframes(data)

    return tmp.read_bytes()


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
