# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system deps (no system python needed — conda provides it)
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

# (optional, but nice) use bash as default shell
SHELL ["/bin/bash", "-lc"]

# -------------------------------
# Create conda env and install deps
# -------------------------------

# Create env with Python 3.10
RUN conda create -n myenv python=3.10 -y

# Install PyTorch (CUDA 12.1 wheels) into that env
RUN /opt/conda/envs/myenv/bin/pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install diffusers from GitHub (latest development version)
RUN /opt/conda/envs/myenv/bin/pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers.git

# Install the rest of your stack into the same env
RUN /opt/conda/envs/myenv/bin/pip install --no-cache-dir \
    transformers \
    protobuf \
    sentencepiece \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pillow \
    "huggingface_hub[cli]" \
    "gguf>=0.10.0" \
    "safetensors" \
    accelerate && \
    /opt/conda/bin/conda clean -afy

# Clean up conda caches
# RUN conda clean -afy

# Expose your API port
EXPOSE 8006

# Hugging Face cache dir (optional but nice)
ENV HF_HOME=/app/.cache/huggingface

# Make the "myenv" env the default Python
ENV CONDA_DEFAULT_ENV=myenv
ENV PATH="/opt/conda/envs/myenv/bin:/opt/conda/bin:${PATH}"

# Copy project files
COPY . .

# Default command
CMD ["python", "flux_service.py"]