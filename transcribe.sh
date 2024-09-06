#!/bin/bash
WHISPER_SERVER_ADDRESS=${WHISPER_SERVER_ADDRESS:-localhost}
WHISPER_PORT=${WHISPER_PORT:-8000}


# Check if a filename is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: ./transcribe.sh <filename>"
  exit 1
fi

# Set the filename from the argument
FILENAME="$1"

# Check if the file exists
if [ ! -f "$FILENAME" ]; then
  echo "File not found: $FILENAME"
  exit 1
fi

# Check if the file has a .wav extension
if [[ "${FILENAME##*.}" != "wav" ]]; then
  echo "Error: Only WAV files are supported."
  exit 1
fi

# Send the WAV file to the Whisper API server and get the transcription
RESPONSE=$(curl -s -X POST "http://$WHISPER_SERVER_ADDRESS:$WHISPER_PORT/transcribe" -H "Content-Type: multipart/form-data" -F "file=@$FILENAME;type=audio/wav")
# Check if the response contains an error message
if [[ "$RESPONSE" == *"detail"* ]]; then
  echo "Error occurred while transcribing the audio file. Response from server:"
  echo "$RESPONSE"
else
  echo "Transcription:"
  echo "$RESPONSE"
fi