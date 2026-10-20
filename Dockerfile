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

# When the container starts, first print the CUDA version that PyTorch sees
# (if torch is installed and built with CUDA), then run the main pipeline.
# This will output something like "CUDA version from torch.version.cuda: 12.1"
# at container startup. If torch is not available or has no CUDA, it will
# print a corresponding message.
CMD ["sh", "-c", "python -c 'import sys;\ntry:\n import torch;\n v = getattr(getattr(torch, \"version\", None), \"cuda\", None);\n print(\"CUDA version from torch.version.cuda:\", v);\nexcept Exception as e:\n print(\"Could not determine CUDA version via torch:\", e);\n sys.exit(0)'; python mnist_models/dvae_pipeline.py"]