from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict
import uuid
import shutil
import tempfile
from utils import process_audio  # Make sure all utility functions are correctly imported

app = FastAPI()

# Initialize the tasks dictionary with type hints to track task status
tasks: Dict[str, Dict] = {}

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...), size_of_model: str = Form(...), task: str = Form(...), source_language: Optional[str] = Form(None), speaker_number: Optional[int] = Form(0)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    session_id = str(uuid.uuid4())  # Generate a unique session ID
    tasks[session_id] = {"status": "processing"}  # Initialize task status

    # Get result from processing
    result = process_audio(temp_file_path, size_of_model, task, source_language, speaker_number)

    # Ensure result is structured correctly
    if 'segments' in result:
        tasks[session_id] = {"status": "completed", "data": result}
    else:
        tasks[session_id] = {"status": "failed", "error": "Invalid data format"}

    # Return JSON with session ID and message
    return JSONResponse(content={"session_id": session_id, "message": "Processing complete"})

@app.get("/task_status/{session_id}")
async def get_task_status(session_id: str):
    task = tasks.get(session_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if 'error' in task:
        return JSONResponse(content={"error": task['error']}, status_code=500)
    return task
