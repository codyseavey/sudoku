# Extraction Service Dockerfile
# Python-based OCR and ML service for sudoku image extraction

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and OCR
# Note: libgl1 replaces libgl1-mesa-glx in newer Debian
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables
ENV HOST=0.0.0.0
ENV PORT=5001
ENV USE_GPU=0

EXPOSE 5001

# Run with Uvicorn (FastAPI)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001", "--timeout-keep-alive", "600"]
