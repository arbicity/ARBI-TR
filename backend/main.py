from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, List
import uuid
import shutil
import tempfile
import os
import json
from utils import process_audio  # Ensure utils.py is adapted for async support if necessary
import asyncio

# Initialize the FastAPI app
app = FastAPI()

# Define global storage for tasks and a queue to manage them
tasks: Dict[str, Dict] = {}
tasks_queue: List[str] = []  # Queue to manage tasks by session_id
queue_lock = asyncio.Lock()  # Asyncio lock for thread-safe operations

async def process_audio_file(task_id: str, background_tasks: BackgroundTasks):
    async with queue_lock:
        if task_id not in tasks_queue:
            return
        task_details = tasks[task_id]
    
    file_path = task_details["file_path"]
    
    try:
        result = process_audio(
            file_path,
            task_details["size_of_model"],
            task_details["task_str"],
            task_details["source_language"],
            task_details["speaker_number"],
        )
        
        tasks[task_id].update({"status": "completed", "data": result})

        # Conditional debug output
        if os.getenv('DEBUG_MODE') == '1':
            print(f"Debug - Task {task_id} completed with result:")
            print(json.dumps(result, indent=2))

    except Exception as e:
        tasks[task_id].update({"status": "failed", "error": str(e)})
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        
        async with queue_lock:
            tasks_queue.remove(task_id)
            
            if tasks_queue:
                next_task_id = tasks_queue[0]
                background_tasks.add_task(process_audio_file, next_task_id, background_tasks)

@app.post("/transcribe/")
async def transcribe_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...), size_of_model: str = Form(...), task_str: str = Form(...), source_language: Optional[str] = Form(None), speaker_number: Optional[int] = Form(0)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        
    temp_file_path = temp_file.name
    
    session_id = str(uuid.uuid4())
    task_info = {
        "file_path": temp_file_path,
        "size_of_model": size_of_model,
        "task_str": task_str,
        "source_language": source_language,
        "speaker_number": speaker_number,
        "status": "queued",
    }
    
    async with queue_lock:
        tasks[session_id] = task_info
        tasks_queue.append(session_id)
    
    if len(tasks_queue) == 1:
        background_tasks.add_task(process_audio_file, session_id, background_tasks)

    return JSONResponse(content={"session_id": session_id, "message": "Your request is queued for processing"})

@app.get("/task_status/{session_id}")
async def get_task_status(session_id: str):
    task = tasks.get(session_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Base response includes status and possibly the queue position.
    task_status = {
        "status": task.get("status"),
        "position": tasks_queue.index(session_id) + 1 if session_id in tasks_queue else None
    }

    # If the task is completed, include the transcription results.
    if task['status'] == 'completed' and 'data' in task:
        task_status['segments'] = task['data']['segments']
    
    return task_status    
