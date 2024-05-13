import os
import json
import ssl
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, List
import uuid
import shutil
import tempfile
from utils import process_audio
import asyncio

app = FastAPI()

tasks: Dict[str, Dict] = {}
tasks_queue: List[str] = []

queue_lock = asyncio.Lock()

async def process_audio_file(task_id: str, background_tasks: BackgroundTasks):
    async with queue_lock:
        if task_id not in tasks_queue:
            return
        task_details = tasks[task_id]

    file_path = task_details["file_path"]
    print(f"Processing file at path: {file_path}, Task ID: {task_id}")
    if os.path.exists(file_path):
        print(f"File details - Size: {os.path.getsize(file_path)} bytes, Exists: Yes")
    else:
        print("File does not exist, check download or file saving process.")

    try:
        result = process_audio(file_path, task_details["size_of_model"], task_details["task_str"], task_details["source_language"], task_details["speaker_number"])
        tasks[task_id].update({"status": "completed", "data": result})
        if os.getenv('DEBUG_MODE') == '1':
            print("Debug - Task completed with result:", json.dumps({"task_id": task_id, "result": result}, indent=2))
    except Exception as e:
        tasks[task_id].update({"status": "failed", "error": str(e)})
        print(f"Error processing file {file_path}: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        async with queue_lock:
            tasks_queue.remove(task_id)
            if tasks_queue:
                background_tasks.add_task(process_audio_file, tasks_queue[0], background_tasks)

@app.post("/transcribe/")
async def transcribe_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...), size_of_model: str = Form(...), task_str: str = Form(...), source_language: Optional[str] = Form(None), speaker_number: Optional[int] = Form(0)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
    temp_file_path = temp_file.name
    session_id = str(uuid.uuid4())
    tasks[session_id] = {"file_path": temp_file_path, "size_of_model": size_of_model, "task_str": task_str, "source_language": source_language, "speaker_number": speaker_number, "status": "queued"}
    async with queue_lock:
        tasks_queue.append(session_id)
    if len(tasks_queue) == 1:
        background_tasks.add_task(process_audio_file, session_id, background_tasks)
    return JSONResponse(content={"session_id": session_id, "message": "Your request is queued for processing"})

@app.get("/task_status/{session_id}")
async def get_task_status(session_id: str):
    task = tasks.get(session_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task_status = {"status": task.get("status"), "position": tasks_queue.index(session_id) + 1 if session_id in tasks_queue else None}
    if task['status'] == 'completed' and 'data' in task:
        task_status['segments'] = task['data']['segments']
    return task_status

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
