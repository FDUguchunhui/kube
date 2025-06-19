FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

WORKDIR /app

# Install uv
RUN pip install uv

# Copy only dependency config first for caching
COPY pyproject.toml .

# Install dependencies (make sure python-dotenv is in pyproject.toml)
RUN uv pip install . --system

# Now copy the rest of the app
COPY . .

# Set visible GPU (optional)
ENV CUDA_VISIBLE_DEVICES=0
# prevent user site packages from being used
ENV PYTHONNOUSERSITE=1
ENV HF_TOKEN=''
ENV CACHE_DIR=''
ENV HF_LOCAL_STORAGE=''

# Do NOT set HF_TOKEN here — pass it via -e at runtime

# Run main script
CMD ["python", "main.py"]

# docker build --platform linux/amd64 -t springlight123/finetune .

# Run command with token:
# docker run --rm --platform linux/amd64 -v $(pwd)/cache:/app/cache springlight123/finetune

# For development:
# docker run --rm --platform linux/amd64 -v $(pwd):/app -v $(pwd)/cache:/app/cache springlight123/finetune
