from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import PlainTextResponse
import whisper
import os
import tempfile
import io
from pyannote.audio import Pipeline

app = FastAPI()

# Load Whisper model
model = whisper.load_model("base")
diarization_pipeline = None

def transcribe_and_diarize(file_path: str, diarization: bool = False) -> str:
    """
    Transcribe audio and optionally perform speaker diarization.

    Args:
        file_path: The path to the audio file.
        diarization: Whether to perform speaker diarization.

    Returns:
        Transcribed text with optional speaker labels.
    """
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

# Define the transcribe_stream endpoint for streamed audio
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
