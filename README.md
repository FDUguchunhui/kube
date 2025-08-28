# LLM Tuning Project

A comprehensive framework for fine-tuning large language models from Hugging Face with support for local development, Docker containerization, and Kubernetes deployment on GPU clusters on MDA Anderson.

## Overview

This project provides a complete pipeline for:
- Loading and preprocessing datasets from Hugging Face
- Fine-tuning pre-trained models from Hugging Face Hub
- Running training jobs locally, in Docker containers, or on Kubernetes clusters
- Automated job generation and deployment for GPU-enabled environments

## Features

- **Hugging Face Integration**: Seamless integration with Hugging Face models, datasets, and tokenizers
- **GPU Support**: Optimized for CUDA-enabled environments with automatic GPU detection
- **Flexible Deployment**: Support for local development, Docker containers, and Kubernetes jobs
- **Caching System**: Efficient caching of models, datasets, and evaluation metrics
- **Job Generation**: Automated Kubernetes job generation with customizable resource allocation
- **Development Tools**: Jupyter notebook for interactive development and testing

## Project Structure

```
llm_tuning/
├── main.py                 # Main training script
├── generate_job.py         # Kubernetes job generator
├── test.ipynb             # Jupyter notebook for development
├── pyproject.toml         # Python project configuration
├── requirements.txt       # Additional Python dependencies
├── dockerfile             # Docker image definition
├── docker-compose.yaml    # Docker Compose configuration
├── configs/
│   └── template/
│       └── kubernetes.yaml # Kubernetes job template
├── kube_jobs/
│   └── train.yaml         # Generated Kubernetes job
└── huggingface_cache/     # Local cache directory
    ├── cache/             # General cache
    ├── datasets/          # Dataset cache
    ├── logs/              # Training logs
    └── models/            # Model cache
```

## Prerequisites

### Environment Requirements
- Python 3.11+
- CUDA-compatible GPU (for training)
- Docker (for containerized deployment)
- Kubernetes cluster with GPU support (for production deployment)

### Required Tokens
- **Hugging Face Token**: Required for accessing models and datasets from Hugging Face Hub
  - Get your token from: https://huggingface.co/settings/tokens



## Installation

### Local Development

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd llm_tuning
   ```

2. **Install dependencies using uv** (recommended):
   ```bash
   pip install uv
   uv pip install . --system
   ```

3. **Set up environment variables**:
You will have to manually create a file name `.env` and fill the following position holder with your actual token
```
HF_TOKEN=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
HOME=/Users/cgu3/Documents/sandbox/llm_tuning
HF_LOCAL_STORAGE=huggingface_cache

API_KEY_HF=hf_xxxxxx
API_KEY_HF_DOWNLOAD=hf_xxxxxx
API_KEY_WANDB=xxxxxx
API_KEY_OPENAI=sk-proj-xxxxxx
WANDB_PROJECT=xxxxxx
WANDB_ENTITY=xxxxxx
CONFIG_PATH=configs/template/kubernetes.yaml

```

### Docker Setup (optional)

You can directly use the built public image directly from Dockerhub. The image serve as a infrastruture to run code that defined somewhere else in you home directory in seadragon. Consider it as your home directory is mounted to that virtual machine and it run your code using pytorch and cuda inside it so you don't have to worry about torch/cuda version. You can change the dockerfile to make new infrastruture to run your code.

1. **Build the Docker image**:
   ```bash
   docker build --platform linux/amd64 -t springlight123/finetune .
   ```

2. **Run with Docker Compose**:
   ```bash
   # Update docker-compose.yaml with your HF_TOKEN
   docker-compose up
   ```

## Usage

### 1. Example Training

Run the training script directly to see a example of llm tuning on your local machine (using cpu).

```bash
python main.py
```

This will:
- Load the GLUE MRPC dataset for text classification
- Download and cache the BERT-base-uncased model
- Fine-tune the model for sequence classification
- Save training logs and outputs to the cache directory

#### Manual Docker Run
You will have to check whether the image is built correctly when try to use it on MDA Kubernetes.
```bash
docker run --rm \
  --platform linux/amd64 \
  -e HF_TOKEN=your_token_here \
  -e HF_LOCAL_STORAGE=huggingface_cache \
  -v $(pwd)/huggingface_cache:/home/huggingface_cache \
  springlight123/finetune
```


### 4. Kubernetes Deployment

#### Generate Kubernetes Job

Use the job generator to create a Kubernetes job configuration:

```bash
python generate_job.py \
  --config configs/template/kubernetes.yaml \
  --hf-token your_token_here \
  --output kube_jobs/my_job.yaml
```

#### Deploy to Kubernetes

```bash
kubectl apply -f kube_jobs/my_job.yaml
```

#### Monitor Job Progress

```bash
# Check job status
kubectl get jobs -n yn-gpu-workload

# View logs
kubectl logs -n yn-gpu-workload job/your-job-name
```

### Kubernetes Job Configuration

Customize the job template in `configs/template/kubernetes.yaml` to change some of the default settings:

```yaml
# Resource requests and limits
resources:
  requests:
    cpu: 12
    memory: 1200Gi
    nvidia.com/gpu: 1
  limits:
    cpu: 12
    memory: 1200Gi
    nvidia.com/gpu: 1

# GPU type selection (optional)
nodeSelector:
  nvidia.com/gpu.present: 'true'
  nvidia.com/gpu.product: 'NVIDIA-H100-80GB-HBM3'  # Specific GPU type
```

## Customization

### Using Different Models

To use a different Hugging Face model, modify `main.py`:

```python
# Example: Using a different BERT variant
checkpoint = "bert-large-uncased"

# Example: Using a different model family
checkpoint = "roberta-base"

# Example: Using a generative model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=model_path)
```


### Troubleshooting

Common issues and solutions:

1. **Out of Memory**: Reduce batch size or model size
2. **Authentication Errors**: Verify HF_TOKEN is set correctly
3. **Model Not Found**: Check model name spelling and availability
4. **Dataset Loading Issues**: Verify dataset name and internet connection

## Advanced Features

### Multi-GPU Training

For multi-GPU setups, modify the Kubernetes configuration:

```yaml
resources:
  requests:
    nvidia.com/gpu: 2  # Request multiple GPUs
  limits:
    nvidia.com/gpu: 2
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Check the troubleshooting section above
- Review Hugging Face documentation: https://huggingface.co/docs
- File an issue in the project repository
