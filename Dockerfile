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

# When the container starts, train all MNIST models. The training script
# will create/update database.db (e.g. mlflow.db) and mnist_models/*_model_mnist_*.
CMD ["python", "mnist_models/train_models.py"]