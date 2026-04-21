# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    build-essential \
    cmake \
    python3-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN grep -v "basicsr" requirements.txt > requirements_no_basicsr.txt && \
    pip3 install --no-cache-dir -r requirements_no_basicsr.txt

# Install basicsr without build isolation so it uses already-installed packages
RUN pip3 install --no-cache-dir --no-build-isolation basicsr==1.4.2

# Install FastAPI and uvicorn for the service
RUN pip3 install --no-cache-dir fastapi uvicorn[standard]

# Copy application files
COPY . .

# Create necessary directories
# RUN mkdir -p output data/erase models/yolov8x

# Download YOLOv8x model if not present
# RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8x.pt')" || true

RUN pip install --no-cache-dir psutil
RUN pip install --no-cache-dir pynvml

# Expose port
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HOME=/app/.cache/huggingface

# Run the service
CMD ["python3", "move_service.py"]
