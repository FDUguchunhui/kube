FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

WORKDIR /home

# Install uv
RUN pip install uv

# Set environment variables early since they rarely change
ENV PYTHONNOUSERSITE=1
ENV CUDA_VISIBLE_DEVICES=0

# Copy only dependency files first
COPY pyproject.toml .

# Install dependencies (make sure python-dotenv is in pyproject.toml)
RUN uv pip install . --system


# These environment variables are likely to change or be overridden at runtime
# so putting them later in the Dockerfile
ENV HF_TOKEN=''
ENV CACHE_DIR=''
ENV HF_LOCAL_STORAGE=''

# Copy source code last since it changes most frequently
# This ensures we only rebuild from this point when code changes
COPY . .

# Run main script
CMD ["python", "llm_tuning/main.py"]

# docker build --platform linux/amd64 -t springlight123/finetune .

# Run command with token:
# docker run --rm --platform linux/amd64 -v $(pwd)/cache:/app/cache springlight123/finetune

# For development:
# docker run --rm --platform linux/amd64 -v $(pwd):/app -v $(pwd)/cache:/app/cache springlight123/finetune
