# Industrial Defect Segmentation System
# Multi-stage Docker build for production deployment

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    QT_QPA_PLATFORM=offscreen

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # PyQt5 dependencies
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Stage 2: Dependencies installation
FROM base AS dependencies

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip3 install --no-cache-dir \
    gunicorn \
    uvicorn \
    python-multipart

# Stage 3: Application
FROM dependencies AS application

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY setup.py .
COPY README.md .

# Install application
RUN pip3 install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p \
    data/raw \
    data/processed/images \
    data/processed/masks \
    data/processed/annotations \
    data/outputs/models \
    data/outputs/predictions \
    data/outputs/reports \
    models/checkpoints \
    logs

# Stage 4: Download SAM weights (optional, can be mounted as volume)
FROM application AS weights

# Download SAM model weights
# Note: This adds ~2.4GB to image size. Consider using volume mount instead.
RUN cd models/checkpoints && \
    wget -q --show-progress \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O sam_vit_h.pth || echo "SAM weights download failed, will need to provide via volume"

# Stage 5: Production image
FROM application AS production

# Copy SAM weights from weights stage (if available)
COPY --from=weights /app/models/checkpoints/*.pth /app/models/checkpoints/ 2>/dev/null || true

# Set proper permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for web interface (if applicable)
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import torch; print('GPU:', torch.cuda.is_available())" || exit 1

# Default command
CMD ["python3", "src/main.py"]

# Stage 6: Development image (includes dev tools)
FROM application AS development

# Install development dependencies
RUN pip3 install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-qt \
    black \
    flake8 \
    mypy \
    isort \
    ipython \
    jupyter

# Install SAM from source for development
RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git

USER appuser

# Override CMD for development
CMD ["python3", "src/main.py", "--debug"]

# Usage:
# 
# Build production image:
#   docker build --target production -t industrial-defect-seg:latest .
#
# Build development image:
#   docker build --target development -t industrial-defect-seg:dev .
#
# Run production container:
#   docker run -d \
#     --name defect-seg \
#     --gpus all \
#     -p 5000:5000 \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     -e DEVICE=cuda \
#     industrial-defect-seg:latest
#
# Run development container:
#   docker run -it --rm \
#     --gpus all \
#     -v $(pwd):/app \
#     -e DEVICE=cuda \
#     industrial-defect-seg:dev bash
