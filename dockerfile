FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files first for better caching
COPY pyproject.toml .

# Install dependencies
RUN uv pip install . --system

# Copy the rest of the application
COPY . .

# Ensure CUDA is available
ENV CUDA_VISIBLE_DEVICES=0

# HF_TOKEN will be passed at runtime
ENV HF_TOKEN=""

# Run the application
CMD ["python", "main.py"]

# Build command:
# docker build --platform linux/amd64 -t pytorch-bert-mrpc .

# Run command with token:
# docker run --rm --platform linux/amd64 -e HF_TOKEN=your_token_here -v $(pwd)/cache:/app/cache pytorch-bert-mrpc

# For development:
# docker run --rm --platform linux/amd64 -e HF_TOKEN=your_token_here -v $(pwd):/app -v $(pwd)/cache:/app/cache pytorch-bert-mrpc
