# Installation Guide

## Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (for GPU training)
- **GPU Memory**: At least 90GB VRAM recommended
- **System Memory**: 32GB+ RAM recommended

## Step-by-Step Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/anonymous/DX-LLaVA.git
cd DX-LLaVA

# Create and activate conda environment
conda create -n dx-llava python=3.10 -y
conda activate dx-llava
```

### 2. Core Dependencies

```bash
# Install the package in development mode
pip install -e .

# Install Flash Attention (recommended for performance)
pip install flash-attn --no-build-isolation
```

### 3. Verify Installation

```bash
# Test basic import
python -c "import llava; print('DX-LLaVA installed successfully!')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Docker Installation

For a containerized environment:

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-devel-ubuntu20.04

RUN apt-get update && apt-get install -y git
COPY . /workspace/DX-LLaVA
WORKDIR /workspace/DX-LLaVA
RUN pip install -e .
```

## Troubleshooting

### Common Issues

1. **Flash Attention Installation Fails**:
   ```bash
   pip install flash-attn --no-build-isolation --no-cache-dir
   ```

2. **CUDA Version Mismatch**:
   Check your CUDA version and install compatible PyTorch:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

3. **Memory Issues**:
   Reduce batch size in training configuration or enable gradient checkpointing.