# --- Stage 1: Builder ---
    # To build for both ARM and x86, use buildx and these commands:
        # docker buildx create --name vla-builder --use
        # docker buildx inspect --bootstrap
        # docker buildx build --platform linux/amd64,linux/arm64 \
        # -t your-docker-username/vla-finetune:latest \
        # --push registry-name .

    # To run:
        # docker run --gpus all \
        # --env-file .env \
        # -v ./outputs:/app/outputs \
        # your-docker-username/vla-finetune:latest

    # We use a full image here to compile any C-extensions if needed
    FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS builder
    
    # Install system dependencies needed for Python and LeRobot (ffmpeg)
    RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        git \
        ffmpeg \
        curl \
        && rm -rf /var/lib/apt/lists/*
    
    # Install uv (The high-performance package manager you're already using)
    COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
    
    WORKDIR /app
    
    # Enable bytecode compilation for faster startup
    ENV UV_COMPILE_BYTECODE=1
    # Link mode "copy" is safer for Docker than "hardlink"
    ENV UV_LINK_MODE=copy
    
    # Copy only dependency files first to leverage Docker's layer caching
    COPY pyproject.toml uv.lock ./
    
    # Install dependencies into a virtual environment
    # This will automatically pull the correct ARM or x86 wheels
    RUN --mount=type=cache,target=/root/.cache/uv \
        uv sync --frozen --no-install-project --no-dev
    
    # --- Stage 2: Runtime ---
    # Use a smaller runtime image for the final container
    FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime
    
    RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        ffmpeg \
        libsm6 \
        libxext6 \
        && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    # Copy the virtual environment from the builder stage
    COPY --from=builder /app/.venv /app/.venv
    
    # Copy your modularized project code
    COPY configs/ ./configs/
    COPY data/ ./data/
    COPY models/ ./models/
    COPY utils/ ./utils/
    COPY train.py .
    
    # Add the virtual environment to the PATH so "python" and "accelerate" just work
    ENV PATH="/app/.venv/bin:$PATH"
    
    # Default command (can be overridden at runtime)
    CMD ["accelerate", "launch", "train.py", "--config", "configs/lora_config.json"]