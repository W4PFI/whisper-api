from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import PlainTextResponse
import whisper
import os

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
    
    # Create an empty buffer to hold the audio stream
    audio_stream = io.BytesIO()
    
    # Read the stream chunk by chunk
    async for chunk in request.stream():
        audio_stream.write(chunk)
    
    # Seek back to the start of the buffer
    audio_stream.seek(0)
    
    # Transcribe audio stream
    result = model.transcribe(audio_stream)

    # Log the transcribed text to Docker logs
    print(f"Transcribed text: {result['text']}")
    
    # Return the transcribed text
    return result["text"]
