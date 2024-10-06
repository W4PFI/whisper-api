from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
import whisper
import os
import tempfile
import io
import threading
import uuid
import datetime
from pyannote.audio import Pipeline

app = FastAPI()
lock = threading.Lock();

# Load Whisper model
model = whisper.load_model("base")
diarization_pipeline = None

def transcribe_and_diarize(file_path: str, diarization: bool = False) -> str:
    """
    Transcribe audio and optionally perform speaker diarization.
    This function is thread-safe and ensures only one transcription and diarization
    process runs at a time using a lock.

    Args:
        file_path: The path to the audio file.
        diarization: Whether to perform speaker diarization.

    Returns:
        Transcribed text with optional speaker labels.
    """
    # Use lock to ensure thread-safety
    with lock:
        # Transcribe the audio once using Whisper
        transcription = model.transcribe(file_path)

        # If diarization is not requested, return the plain transcription
        if not diarization:
            return transcription["text"]

        # Initialize diarization pipeline only when needed
        global diarization_pipeline
        if diarization_pipeline is None:
            huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
            if not huggingface_token:
                raise RuntimeError("HUGGINGFACE_TOKEN environment variable is not set")

            # Load pyannote speaker diarization pipeline with the token
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization", 
                use_auth_token=huggingface_token
            )

        # Perform speaker diarization
        diarization_result = diarization_pipeline(file_path)

        # Combine transcription with speaker information
        result = []
        for segment in transcription['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']

            # Find the corresponding speaker based on time
            speaker_label = "Unknown Speaker"
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                if turn.start <= start_time <= turn.end:
                    speaker_label = speaker
                    break

            # Append the result with the speaker label
            result.append(f"Speaker {speaker_label}: {text}")

        return "\n".join(result)

def process_transcription_background(file_path: str, transcript_file_path: str, diarization: bool):
    """
    Background task to transcribe and save the result to a file.

    Args:
        file_path: Path to the uploaded audio file.
        transcript_file_path: Path to save the transcription.
        diarization: Whether to use speaker diarization.
    """
    try:
        # Perform transcription and optionally diarization
        transcribed_text = transcribe_and_diarize(file_path, diarization)

        # Save the transcription result
        with open(transcript_file_path, "w") as transcript_file:
            transcript_file.write(transcribed_text)
        
        print(f"Transcription saved to {transcript_file_path}")

    finally:
        # Clean up the audio file
        os.remove(file_path)


# Define the transcribe endpoint for file uploads
@app.post("/transcribe", response_class=PlainTextResponse)
async def transcribe_audio(file: UploadFile = File(...), diarization: bool = Query(False)):
    if file.content_type not in ["audio/wav", "audio/x-wav"]:
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")
    
    # Save uploaded file to disk
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Perform transcription and optionally diarization
    transcribed_text = transcribe_and_diarize(file_path, diarization=diarization)
    print(f"Transcribed text: {transcribed_text}")

    # Delete the file after processing
    os.remove(file_path)

    return transcribed_text

# Endpoint for streaming audio transcription
# 
# This endpoint receives an audio stream (WAV format) as input, reads the stream 
# chunk by chunk, and writes it to a temporary file. The Whisper model is used to 
# transcribe the audio, with an optional feature to identify different speakers 
# using speaker diarization. The final transcription, including speaker labels 
# if requested, is returned to the client.
@app.post("/transcribe_stream", response_class=PlainTextResponse)
async def transcribe_audio_stream(request: Request, diarization: bool = Query(False)):
    if request.headers.get('content-type') not in ["audio/wav", "audio/x-wav"]:
        raise HTTPException(status_code=400, detail="Only WAV streams are supported.")
    
    audio_stream = io.BytesIO()
    async for chunk in request.stream():
        audio_stream.write(chunk)
    
    audio_stream.seek(0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_stream.getvalue())
        temp_file_path = temp_audio_file.name

    # Perform transcription and optionally diarization
    transcribed_text = transcribe_and_diarize(temp_file_path, diarization=diarization)
    print(f"Transcribed text: {transcribed_text}")
    os.remove(temp_file_path)

    return transcribed_text

# Background transcription endpoint for streamed audio
#
# This endpoint accepts an audio stream (WAV format) and writes it to a temporary file.
# Unlike the regular transcribe endpoint, it immediately returns a success message to the client 
# without waiting for the transcription process to complete. The transcription, including optional 
# speaker diarization, runs asynchronously in the background, and the resulting text is saved 
# to a specified directory, defined by the TRANSCRIPT_DIR environment variable.
@app.post("/transcribe_stream_offline")
async def transcribe_audio_offline(request: Request, background_tasks: BackgroundTasks, diarization: bool = Query(False)):
    if request.headers.get('content-type') not in ["audio/wav", "audio/x-wav"]:
        raise HTTPException(status_code=400, detail="Only WAV streams are supported.")

    audio_stream = io.BytesIO()
    async for chunk in request.stream():
        audio_stream.write(chunk)

    audio_stream.seek(0)

    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_stream.getvalue())
        temp_file_path = temp_audio_file.name

    # Define the transcript output path based on environment variable TRANSCRIPT_DIR
    transcript_dir = os.getenv("TRANSCRIPT_DIR", "/tmp")
    
    # Generate the current timestamp in the desired format
    timestamp = datetime.now().strftime("%m%d%Y%H%M%S")
    
    # Generate a unique filename using the timestamp and UUID
    transcript_filename = f"transcript_{timestamp}_{uuid.uuid4()}.txt"
    transcript_file_path = os.path.join(transcript_dir, transcript_filename)

    # Schedule background transcription
    background_tasks.add_task(process_transcription_background, temp_file_path, transcript_file_path, diarization)

    return {"detail": "Transcription is being processed in the background."}

