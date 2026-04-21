# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system deps (no python needed — conda provides it)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Add conda to PATH so we can use it
ENV PATH="/opt/conda/bin:${PATH}"

# Accept Anaconda Terms of Service for defaults channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Copy environment file and create the conda env
COPY objectclearEnv.yml .
RUN conda env create -f objectclearEnv.yml

# Use bash as default shell for later RUNs (optional but nice)
SHELL ["bash", "-lc"]

# Activate environment by default (via PATH)
ENV CONDA_DEFAULT_ENV=objectclear
ENV PATH="/opt/conda/envs/objectclear/bin:${PATH}"

# Install FastAPI + uvicorn inside the conda env
RUN pip install --no-cache-dir fastapi uvicorn[standard]
# Install the missing dependency for OpenGL
RUN apt-get update && apt-get install -y libgl1-mesa-glx
# Add this command to your Dockerfile before running your application
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir psutil

# Copy project files
COPY . .

# Create output directory
RUN mkdir -p output logs

# Expose port
EXPOSE 8004

# Disable python buffering and set CUDA / HF cache
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HOME=/app/.cache/huggingface

# Run service using the conda env python
CMD ["python", "objectclearservice.py"]
