FROM python:3.12-slim

# Avoid prompts from apt and ensure UTF-8
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies required by scientific Python stack and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirement spec first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the whole repo into the image (respecting .dockerignore)
COPY . .

# When the container starts, first check that a GPU is visible inside the
# container (i.e. it was started with `--gpus` and NVIDIA Docker runtime).
# If no GPU is detected, exit with a clear error instead of failing later
# with a PyTorch / CUDA error.
CMD ["echo", "test"]
CMD ["echo", "nvidia-smi"]
CMD ["sh", "-lc", "if [ -z \"$NVIDIA_VISIBLE_DEVICES\" ] || [ \"$NVIDIA_VISIBLE_DEVICES\" = \"void\" ]; then echo 'ERROR: No GPU detected inside the container.'; echo '       Please run the container with GPU support, e.g.:'; echo '         docker run --rm --gpus all <image_name>'; exit 1; fi; python mnist_models/dvae_pipeline.py"]