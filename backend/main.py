import os
import json
import ssl
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyQuery
from fastapi.security import APIKeyHeader
from typing import Optional, Dict, List
import uuid
import shutil
import tempfile
from utils import process_audio, load_audio_file
import asyncio

app = FastAPI()

api_key_security = APIKeyQuery(name="api_key", auto_error=False)

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


# API keys and their limits in minutes
api_keys: Dict[str, float] = {"key1": 180.0}
processed_minutes: Dict[str, float] = {}

def validate_api_key(api_key: str = Security(api_key_header)):
    # strip the 'Bearer ' prefix if you're using Bearer tokens
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]
    if api_key not in api_keys:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    if api_key in processed_minutes and processed_minutes[api_key] >= api_keys[api_key]:
        raise HTTPException(status_code=429, detail="API Key quota exceeded")
    return api_key

tasks: Dict[str, Dict] = {}
tasks_queue: List[str] = []

queue_lock = asyncio.Lock()

async def process_audio_file(task_id: str, background_tasks: BackgroundTasks):
    async with queue_lock:
        if task_id not in tasks_queue:
            return
        task_details = tasks[task_id]

    file_path = task_details["file_path"]
    try:
        duration, result = process_audio(file_path, task_details["size_of_model"], task_details["task_str"], task_details["source_language"], task_details["speaker_number"])
        tasks[task_id].update({"status": "completed", "data": result})
        update_processed_minutes(task_details["api_key"], duration)
    except Exception as e:
        tasks[task_id].update({"status": "failed", "error": str(e)})
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        async with queue_lock:
            tasks_queue.remove(task_id)
            if tasks_queue:
                background_tasks.add_task(process_audio_file, tasks_queue[0], background_tasks)

@app.post("/transcribe/")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    size_of_model: str = Form(...),
    task_str: str = Form(...),
    source_language: Optional[str] = Form(None),
    speaker_number: Optional[int] = Form(0),
    api_key: str = Security(validate_api_key)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
    temp_file_path = temp_file.name
    session_id = str(uuid.uuid4())
    tasks[session_id] = {
        "file_path": temp_file_path, 
        "size_of_model": size_of_model, 
        "task_str": task_str, 
        "source_language": source_language, 
        "speaker_number": speaker_number, 
        "status": "queued", 
        "api_key": api_key
    }
    async with queue_lock:
        tasks_queue.append(session_id)
    if len(tasks_queue) == 1:
        background_tasks.add_task(process_audio_file, session_id, background_tasks)
    return JSONResponse(content={"session_id": session_id, "message": "Your request is queued for processing"})

@app.get("/task_status/{session_id}")
async def get_task_status(session_id: str, api_key: str = Security(validate_api_key)):
    task = tasks.get(session_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    # Check if the api_key in the task matches the one provided
    if task['api_key'] != api_key:
        raise HTTPException(status_code=403, detail="Unauthorized access to task status")
    
    task_status = {"status": task.get("status"), "position": tasks_queue.index(session_id) + 1 if session_id in tasks_queue else None}
    if task['status'] == 'completed' and 'data' in task:
        task_status['segments'] = task['data']['segments']
    return task_status

def update_processed_minutes(api_key: str, duration_in_seconds: float):
    if api_key in processed_minutes:
        processed_minutes[api_key] += duration_in_seconds / 60
    else:
        processed_minutes[api_key] = duration_in_seconds / 60

if __name__ == "__main__":
    use_mtls = os.getenv('USE_MTLS') == '1'
    host = "0.0.0.0"
    port = 8000

    if use_mtls:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain('/app/certs/signed_client.crt', '/app/certs/client.key')
        ssl_context.load_verify_locations('/app/certs/ca.crt')
        ssl_context.verify_mode = ssl.CERT_REQUIRED

        config = uvicorn.Config(app=app, host=host, port=port, ssl_keyfile='/app/certs/client.key', ssl_certfile='/app/certs/signed_client.crt')
        server = uvicorn.Server(config)
        server.run()
    else:
        # Run without SSL context if mTLS is not enabled
        uvicorn.run(app, host=host, port=port)
