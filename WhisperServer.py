from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import PlainTextResponse
import whisper
import os
import tempfile
import io

app = FastAPI()

# Load Whisper model
model = whisper.load_model("base")

@app.post("/transcribe", response_class=PlainTextResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    # Check if the uploaded file is an audio file
    if file.content_type not in ["audio/wav", "audio/x-wav"]:
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")
    
    # Save uploaded file to disk
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Transcribe audio file
    result = model.transcribe(file_path) 

    # Log the transcribed text to Docker logs
    print(f"Transcribed text: {result['text']}")

    # Delete the file after processing
    # os.remove(file_path)

    # Return the transcribed text
    return result["text"]

@app.post("/transcribe_stream", response_class=PlainTextResponse)
async def transcribe_audio_stream(request: Request):
    # Check if the content type is correct
    if request.headers.get('content-type') not in ["audio/wav", "audio/x-wav"]:
        raise HTTPException(status_code=400, detail="Only WAV streams are supported.")
    
    # Create an empty buffer to hold the streamed audio data
    audio_stream = io.BytesIO()
    
    # Read the stream chunk by chunk and write to the buffer
    async for chunk in request.stream():
        audio_stream.write(chunk)
    
    # Seek back to the start of the buffer
    audio_stream.seek(0)

    # Create a temporary file using `tempfile`
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        # Write the streamed audio to the temp file
        temp_audio_file.write(audio_stream.getvalue())
        temp_file_path = temp_audio_file.name

    # Optional: Log or verify the temp file creation
    file_size = os.path.getsize(temp_file_path)
    print(f"Saved temp audio file at {temp_file_path}, size: {file_size} bytes")

    # Transcribe the saved temp WAV file using Whisper
    result = model.transcribe(temp_file_path)

    # Log the transcribed text to Docker logs
    print(f"Transcribed text: {result['text']}")

    # Clean up: Delete the temporary file after processing
    # os.remove(temp_file_path)

    # Return the transcribed text
    return result["text"]
