# Build from the repository root:  docker build -f docker/Dockerfile -t dtmr .
# Base image: CUDA 12.6 on Ubuntu 22.04 (MuJoCo Warp needs an NVIDIA GPU + driver on the host)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3.10-venv \
        git \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python/pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade pip

WORKDIR /workspace

# Copy project and install (editable mode so mounted changes take effect)
COPY . .

RUN pip install --no-cache-dir -e .

# Default to an interactive shell
CMD ["bash"]
