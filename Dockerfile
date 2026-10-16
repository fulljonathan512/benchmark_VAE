FROM python:3.10-slim

# Avoid prompts from apt and ensure UTF-8
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps needed for scipy, torch, torchvision, mlflow, pytorch-fid, etc.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     git \
#     curl \
#     libglib2.0-0 \
#     libsm6 \
#     libxrender1 \
#     libxext6 \
#     libjpeg-dev \
#     zlib1g-dev \
#     && rm -rf /var/lib/apt/lists/*

# Copy only requirement spec first for better caching if you later add more deps
# COPY requirements.txt ./

# Install base project requirements
# RUN pip install --no-cache-dir --upgrade pip && \
    # pip install --no-cache-dir -r requirements.txt

# Additional runtime dependencies used in mnist_models/evaluate_mnist_models.py
# (pythae, mlflow, pytorch-fid, torchvision, etc.)
# RUN pip install --no-cache-dir \
#     torchvision \
#     mlflow \
#     pythae \
#     pytorch-fid

# Copy the whole repo (including existing models + mlflow.db if present)
# COPY . .

# Default command:
#  1. Train all MNIST models (creates/updates *_model_mnist_* directories)
#  2. Evaluate them and log metrics to mlflow.db in the repo root
# CMD ["/bin/sh", "-c", "\
# cd /app && \
# echo '===== TRAIN MNIST MODELS =====' && \
# python mnist_models/train_models.py && \
# echo '===== EVALUATE MNIST MODELS =====' && \
# python mnist_models/evaluate_mnist_models.py\
# "]
CMD ["echo", "====test===="]