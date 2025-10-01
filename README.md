# PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection

This repository provides the official implementation of **PRPO** (Paragraph-level Relative Policy Optimization), a reinforcement learning framework designed to improve reasoning and detection accuracy in vision-language deepfake detection tasks.

## Overview

This folder contains the DX-LLaVA implementation for fine-tuning LLaVA (Large Language and Vision Assistant) with ConvNeXt vision encoder for deepfake detection tasks. DX-LLAVA enhances the standard LLaVA architecture by integrating ConvNeXt as the vision encoder, specifically optimized for visual deception detection. The model follows a two-stage training process: feature alignment pretraining and visual instruction tuning.

## Prerequisites

- CUDA-capable GPU (recommended: 2x H100 94GB or similar)
- Python 3.10+
- Conda/Miniconda

## Setup

### 1. Environment Setup

Create and activate the conda environment using the provided configuration:

```bash
conda env create -f llava.yaml
conda activate llava
```

### 2. Dataset Preparation

To access the DF-R5 dataset, please fill out the request form:

[DF-R5 Dataset Access Form](https://docs.google.com/forms/d/e/1FAIpQLSf4RCDkaWzCyx_X4-OJ1-h0ICyraiSGFamZ_zuC7Kc9B2FPug/viewform?usp=header)

Your dataset should be organized as follows:
```
DF_R5_processed_images/
  ├── [domain name]/
  │   ├── train/
  │   │   ├── fake/
  │   │   └── real/
  │   ├── val/
  │   │   ├── fake/
  │   │   └── real/
  │   └── test/
  │       ├── fake/
  │       └── real/

```

### 3. Model Checkpoints

Ensure you have the base model checkpoints available:
- Pre-trained LLaVA checkpoint
- ConvNeXt vision encoder weights

## Training

```bash
./run_train_dx_llava.sh
```

### Configuration

Key training parameters can be modified in `run_train_dx_llava.sh`:

```bash
# Model and data paths
MODEL_PATH="../checkpoints/dx-llava-binary-ddim"
IMAGE_FOLDER="../DF_R5_processed_images"
TRAIN_DATA="./data/train.json"

# Training hyperparameters
BATCH_SIZE=8
LEARNING_RATE=2e-5
NUM_EPOCHS=3
MAX_LENGTH=2048
```

## Evaluation

### Quick Start

Evaluate your trained model:

```bash
./run_eval_dx_llava.sh
```

### Evaluation Process

The evaluation script includes:

1. **Question Generation**: Creates evaluation questions from dataset
2. **Model Inference**: Runs multi-GPU inference for efficiency
3. **Performance Metrics**: Calculates accuracy, precision, recall, and F1-score

### Configuration

Modify evaluation parameters in `run_eval_dx_llava.sh`:

```bash
# Evaluation settings
MODEL_PATH="../checkpoints/dx-llava-binary-ddim"
TEST_DATA="./question/df_question_ddim_1000.jsonl"
OUTPUT_FILE="./answer/dx_llava_eval_results.jsonl"
BATCH_SIZE=32
```

## Model Architecture

DX-LLAVA incorporates:

- **Vision Encoder**: ConvNeXt-Large for robust visual feature extraction
- **Vision-Language Connector**: Two-layer MLP projection
- **Language Model**: Vicuna/LLaMA-based backbone
- **Specialized Head**: Binary classification for deepfake detection

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **DataLoader Errors**: Verify dataset paths and format
3. **Checkpoint Loading**: Ensure model paths are correct and accessible

### Memory Optimization

For limited GPU memory:

```bash
# Use gradient checkpointing and mixed precision
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{prpo2025tuan,
   title={PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection}, 
   author={Tuan Nguyen and Naseem Khan and Khang Tran and NhatHai Phan and Issa Khalil},
   year={2025},
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- [Original LLaVA implementation](https://github.com/haotian-liu/LLaVA)
- ConvNeXt architecture
- DeepSpeed optimization framework