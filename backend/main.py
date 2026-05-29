import asyncio
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

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


async def _run_task(task_id: str, background_tasks: BackgroundTasks) -> None:
    """Process the next queued task, then kick off the one after it."""
    async with queue_lock:
        if task_id not in tasks_queue:
            return
        task_details = tasks[task_id]

    file_path = task_details["file_path"]
    logger.info(f"Processing task {task_id}: {file_path}")

    try:
        result = await asyncio.to_thread(
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
    queue_length: int


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


class TaskStatusResponse(BaseModel):
    status: str
    position: Optional[int] = None
    segments: Optional[List[Segment]] = None
    error: Optional[str] = None


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", queue_length=len(tasks_queue))


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

    session_id = str(uuid.uuid4())
    tasks[session_id] = {
        "file_path": tmp_path,
        "task_str": task_str,
        "source_language": source_language,
        "speaker_number": speaker_number,
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
    )


@app.get("/task_status/{session_id}")
async def get_task_status(session_id: str) -> TaskStatusResponse:
    task = tasks.get(session_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    position = tasks_queue.index(session_id) + 1 if session_id in tasks_queue else None
    segments = [Segment(**s) for s in task["data"]["segments"]] if task["status"] == "completed" else None
    error = task.get("error") if task["status"] == "failed" else None
    return TaskStatusResponse(status=task["status"], position=position, segments=segments, error=error)


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints
# ---------------------------------------------------------------------------


class Transcription(BaseModel):
    text: str


@app.post("/v1/audio/transcriptions", response_model=Transcription)
async def transcribe_openai(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
) -> Transcription:
    suffix = os.path.splitext(file.filename or "audio")[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = await asyncio.to_thread(process_audio_without_diarization, tmp_path, "transcribe", language)
        return Transcription(text=result["text"])
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
    suffix = os.path.splitext(file.filename or "audio")[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = await asyncio.to_thread(process_audio_without_diarization, tmp_path, "translate", None)
        return Transcription(text=result["text"])
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
