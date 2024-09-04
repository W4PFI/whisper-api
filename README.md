## Whisper API

### Overview

This project sets up a Docker container running OpenAI's Whisper model for speech-to-text. It provides a FastAPI-based API to upload a WAV file and receive a text transcription.

### Files

- **Dockerfile**: Defines the Docker environment.
- **WhisperServer.py**: FastAPI server handling file uploads and transcribing audio files.
- **requirements.txt**: Lists the Python dependencies.
- **startDocker.sh**: Bash script to build and start the Docker container.
- **stopDocker.sh**: Bash script to stop and remove the Docker container.

### Usage

#### 1. Start the Docker Container

Run the following command to build and start the Docker container:

```bash
./startDocker.sh
