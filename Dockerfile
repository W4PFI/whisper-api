# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables (optional) - required for diarization (speaker lables)
# You can specify the token in the Dockerfile, but it's more secure to pass it at runtime.
# ENV HUGGINGFACE_TOKEN your_token_here

# Command to run the FastAPI server
CMD ["uvicorn", "WhisperServer:app", "--host", "0.0.0.0", "--port", "8000"]
