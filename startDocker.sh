#!/bin/bash

# Build the Docker image
docker build -t whisper-api .

# Run the Docker container
docker run -d --name whisper-api -p 8001:8000 -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN -e TRANSCRIPT_DIR=$DATA_DIR whisper-api
