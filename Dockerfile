# Base image: Python 3.10 on Linux (slim = smaller size)
FROM python:3.10-slim

# Install system dependencies needed by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install Python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the V6 folder into the container
COPY versions/v6-popsign-hands-only/ ./versions/v6-popsign-hands-only/

# Set the working directory to where the script lives
WORKDIR /app/versions/v6-popsign-hands-only/scripts

# Default command to run the inference system
CMD ["python", "test_webcam_sentences.py"]