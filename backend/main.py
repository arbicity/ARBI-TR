import asyncio
import hashlib
import os
import shutil
import subprocess  # nosec B404 — fixed ffprobe argv, no shell
import tempfile
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel
from utils import initialize_models, process_audio, process_audio_without_diarization

# ---------------------------------------------------------------------------
# App lifecycle — models loaded once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ARBI-TR — loading models...")
    await asyncio.to_thread(initialize_models)
    logger.info("Models ready")
    yield
    logger.info("Shutting down ARBI-TR")


app = FastAPI(title="ARBI-TR", version="1.0.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# In-memory task queue (single-process; replace with Redis Streams for multi-worker)
# ---------------------------------------------------------------------------

tasks: Dict[str, Dict] = {}
tasks_queue: List[str] = []
queue_lock = asyncio.Lock()

# Serialize ALL GPU work (async queue worker + synchronous endpoints) so a single
# GPU is never asked to run two transcription/diarization jobs at once.
_gpu_sema = asyncio.Semaphore(1)


async def _run_gpu(fn, *args):
    """Run a (blocking) GPU pipeline function in a thread, one job at a time."""
    async with _gpu_sema:
        return await asyncio.to_thread(fn, *args)


# ---------------------------------------------------------------------------
# Sync-endpoint robustness: back-pressure + content-addressed result cache.
#
# The OpenAI sync endpoints hold the HTTP connection for the whole job (this is
# how OpenAI's own transcription API works, and it's what lets a gateway bill
# per-second off the response duration). Two safeguards make that model robust:
#
#   * back-pressure — once _MAX_INFLIGHT jobs are queued/running on the GPU,
#     new callers get 503 + Retry-After instead of piling up held connections.
#   * result cache — keyed by (audio bytes + params). A cache hit skips the GPU
#     entirely, so a proxy/client that times out and RETRIES a long job gets the
#     already-computed transcript instantly (and is billed once, since the
#     timed-out first attempt never returned a billable response). Concurrent
#     identical requests coalesce onto the single in-flight job.
# ---------------------------------------------------------------------------

_MAX_INFLIGHT = int(os.getenv("ARBI_TR_MAX_INFLIGHT", "32"))
_RESULT_TTL_S = float(os.getenv("ARBI_TR_RESULT_TTL_S", "3600"))
_RESULT_CACHE_MAX = int(os.getenv("ARBI_TR_RESULT_CACHE_MAX", "256"))

_pending = 0  # jobs past the back-pressure gate (waiting on or holding the GPU)
_pending_lock = asyncio.Lock()
_inflight: Dict[str, asyncio.Future] = {}  # content-key -> in-flight GPU job
_result_cache: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()  # key -> (expiry_ts, result)
_cache_lock = asyncio.Lock()


def _hash_file(path: str, *parts: object) -> str:
    """Content-address an upload: sha256 of the bytes plus the params that
    change the transcript (task, diarization, language)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    for part in parts:
        h.update(b"\x00")
        h.update(str(part).encode())
    return h.hexdigest()


@asynccontextmanager
async def _backpressure():
    """Reject (503) once too many jobs are queued/running on the GPU."""
    global _pending
    async with _pending_lock:
        if _pending >= _MAX_INFLIGHT:
            raise HTTPException(
                status_code=503,
                detail="Transcription queue full; retry shortly",
                headers={"Retry-After": "30"},
            )
        _pending += 1
    try:
        yield
    finally:
        async with _pending_lock:
            _pending -= 1


async def _cache_get(key: str) -> Optional[Any]:
    async with _cache_lock:
        hit = _result_cache.get(key)
        if hit is None:
            return None
        expiry, result = hit
        if expiry <= time.time():
            _result_cache.pop(key, None)
            return None
        _result_cache.move_to_end(key)
        return result


async def _cache_put(key: str, result: Any) -> None:
    async with _cache_lock:
        now = time.time()
        _result_cache[key] = (now + _RESULT_TTL_S, result)
        _result_cache.move_to_end(key)
        for k in [k for k, (exp, _) in list(_result_cache.items()) if exp <= now]:
            _result_cache.pop(k, None)
        while len(_result_cache) > _RESULT_CACHE_MAX:
            _result_cache.popitem(last=False)


async def _run_gpu_cached(key: str, fn, *args) -> Any:
    """Run a GPU job at most once per content key. Cache hits skip the GPU;
    concurrent identical requests coalesce onto one job. Back-pressure is applied
    only to the owner that actually runs the GPU (hits/coalesced waiters are free)."""
    cached = await _cache_get(key)
    if cached is not None:
        return cached

    async with _cache_lock:
        fut = _inflight.get(key)
        owner = fut is None
        if owner:
            fut = asyncio.get_running_loop().create_future()
            # Always "retrieve" the outcome so a failure with no coalesced
            # waiters doesn't log "Future exception was never retrieved".
            fut.add_done_callback(lambda f: f.cancelled() or f.exception())
            _inflight[key] = fut

    if not owner:
        return await fut  # someone else is already transcribing this exact input

    try:
        async with _backpressure():
            result = await _run_gpu(fn, *args)
    except BaseException as exc:
        async with _cache_lock:
            _inflight.pop(key, None)
        if not fut.done():
            fut.set_exception(exc)
        raise

    await _cache_put(key, result)
    async with _cache_lock:
        _inflight.pop(key, None)
    if not fut.done():
        fut.set_result(result)
    return result


def _probe_duration(path: str) -> float:
    """Audio/video duration in seconds via ffprobe (metadata read, no decode).

    Returns 0.0 if ffprobe is unavailable or the container has no duration —
    callers treat a non-positive duration as "unknown" for billing.
    """
    try:
        out = subprocess.run(  # nosec B603 — fixed argv, no shell, trusted binary
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return float(out.stdout.strip() or 0.0)
    except Exception as exc:  # noqa: BLE001 — duration is best-effort
        logger.warning(f"ffprobe failed for {path}: {exc}")
        return 0.0


async def _run_task(task_id: str, background_tasks: BackgroundTasks) -> None:
    """Process the next queued task, then kick off the one after it."""
    async with queue_lock:
        if task_id not in tasks_queue:
            return
        task_details = tasks[task_id]

    file_path = task_details["file_path"]
    logger.info(f"Processing task {task_id}: {file_path}")

    try:
        result = await _run_gpu(
            process_audio,
            file_path,
            task_details["task_str"],
            task_details["source_language"],
            task_details["speaker_number"],
        )
        tasks[task_id].update({"status": "completed", "data": result})
        logger.info(f"Task {task_id} completed with {len(result.get('segments', []))} segments")
    except Exception as exc:
        tasks[task_id].update({"status": "failed", "error": str(exc)})
        logger.exception(f"Task {task_id} failed: {exc}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        async with queue_lock:
            if task_id in tasks_queue:
                tasks_queue.remove(task_id)
            if tasks_queue:
                background_tasks.add_task(_run_task, tasks_queue[0], background_tasks)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    queue_length: int  # async /transcribe/ queue
    inflight: int = 0  # sync-endpoint jobs queued/running on the GPU
    max_inflight: int = 0  # back-pressure ceiling (503 beyond this)


class Segment(BaseModel):
    """A diarized, transcribed span. Timestamps are HH:MM:SS strings."""

    Start: str
    End: str
    Speaker: str
    Text: str


class TranscribeSubmitResponse(BaseModel):
    session_id: str
    message: str
    queue_position: int
    # Audio duration (seconds), probed at submit before any GPU work. Lets a
    # gateway bill by audio length immediately, decoupled from queue/GPU time.
    duration: Optional[float] = None


class TaskStatusResponse(BaseModel):
    status: str
    position: Optional[int] = None
    segments: Optional[List[Segment]] = None
    error: Optional[str] = None
    duration: Optional[float] = None


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        queue_length=len(tasks_queue),
        inflight=_pending,
        max_inflight=_MAX_INFLIGHT,
    )


@app.post("/transcribe/")
async def transcribe_audio_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    size_of_model: str = Form(...),  # kept for API compatibility, ignored internally
    task_str: str = Form(...),
    source_language: Optional[str] = Form(None),
    speaker_number: Optional[int] = Form(0),
) -> TranscribeSubmitResponse:
    suffix = os.path.splitext(file.filename or "audio")[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    duration = await asyncio.to_thread(_probe_duration, tmp_path)

    session_id = str(uuid.uuid4())
    tasks[session_id] = {
        "file_path": tmp_path,
        "task_str": task_str,
        "source_language": source_language,
        "speaker_number": speaker_number,
        "duration": duration,
        "status": "queued",
    }
    async with queue_lock:
        tasks_queue.append(session_id)

    if len(tasks_queue) == 1:
        background_tasks.add_task(_run_task, session_id, background_tasks)

    return TranscribeSubmitResponse(
        session_id=session_id,
        message="Your request is queued for processing",
        queue_position=len(tasks_queue),
        duration=duration,
    )


@app.get("/task_status/{session_id}")
async def get_task_status(session_id: str) -> TaskStatusResponse:
    task = tasks.get(session_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    position = tasks_queue.index(session_id) + 1 if session_id in tasks_queue else None
    segments = [Segment(**s) for s in task["data"]["segments"]] if task["status"] == "completed" else None
    error = task.get("error") if task["status"] == "failed" else None
    return TaskStatusResponse(
        status=task["status"],
        position=position,
        segments=segments,
        error=error,
        duration=task.get("duration"),
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints
# ---------------------------------------------------------------------------


class Transcription(BaseModel):
    """OpenAI `response_format=json` transcription response. `duration` (audio
    seconds) is an ARBI-TR extension so gateways can bill by audio length."""

    text: str
    duration: Optional[float] = None


class TranscriptionSegment(BaseModel):
    """A segment in an OpenAI `verbose_json` response. `speaker` is an ARBI-TR
    extension, populated only for diarization models."""

    id: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class VerboseTranscription(BaseModel):
    """OpenAI `response_format=verbose_json` transcription response.

    `duration` and `segments` are required (and absent from the plain
    `Transcription` shape) so the two response models are unambiguous to
    generated/typed clients parsing the union.
    """

    text: str
    duration: float
    segments: List[TranscriptionSegment]
    task: str = "transcribe"
    language: Optional[str] = None


def _hms_to_seconds(hms: str) -> float:
    """Parse a 'H:MM:SS' timestamp (from convert_time) into float seconds."""
    try:
        parts = hms.split(".")[0].split(":")
        h, m, s = ([0, 0, 0] + [int(p) for p in parts])[-3:]
        return float(h * 3600 + m * 60 + s)
    except Exception:
        return 0.0


def _save_upload(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename or "audio")[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


@app.post("/v1/audio/transcriptions")
async def transcribe_openai(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
) -> VerboseTranscription | Transcription:
    """OpenAI-compatible, synchronous transcription.

    Plain transcription by default. When `model` selects a diarization variant
    (name contains "diarize"), runs the full speaker-diarization pipeline
    synchronously and — with `response_format=verbose_json` — returns
    per-segment speaker labels. Routable/billable through an OpenAI gateway
    (e.g. LiteLLM); the diarization tier is chosen by model name since gateways
    drop non-standard params.
    """
    diarize = "diarize" in model.lower()
    tmp_path = _save_upload(file)
    try:
        # Key the cached GPU result on the inputs that change the transcript
        # (NOT response_format — that only reshapes the same result below).
        key = _hash_file(tmp_path, "transcribe", "diarize" if diarize else "plain", language or "")
        if diarize:
            result = await _run_gpu_cached(key, process_audio, tmp_path, "transcribe", language, 0)
            raw = result.get("segments", [])
            segments = [
                TranscriptionSegment(
                    id=i,
                    start=_hms_to_seconds(s["Start"]),
                    end=_hms_to_seconds(s["End"]),
                    text=s["Text"],
                    speaker=s.get("Speaker"),
                )
                for i, s in enumerate(raw)
            ]
            text = " ".join(s["Text"] for s in raw).strip()
            duration = max((seg.end for seg in segments), default=0.0)
            if response_format == "verbose_json":
                return VerboseTranscription(language=language, duration=duration, text=text, segments=segments)
            return Transcription(text=text, duration=duration)

        result = await _run_gpu_cached(key, process_audio_without_diarization, tmp_path, "transcribe", language)
        text = result["text"]
        duration = float(result.get("duration") or 0.0)
        if response_format == "verbose_json":
            return VerboseTranscription(language=language, text=text, duration=duration, segments=[])
        return Transcription(text=text, duration=duration)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/v1/audio/translations", response_model=Transcription)
async def translate_openai(
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
) -> Transcription:
    tmp_path = _save_upload(file)
    try:
        key = _hash_file(tmp_path, "translate")
        result = await _run_gpu_cached(key, process_audio_without_diarization, tmp_path, "translate", None)
        return Transcription(text=result["text"], duration=float(result.get("duration") or 0.0))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
