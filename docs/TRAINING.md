# Training Guide

## Overview

DX-LLaVA training involves fine-tuning the LLaVA model with ConvNeXt vision encoder for deepfake detection tasks.

## Training Configuration

### Basic Training Command

```bash
./train_dx_llava.sh
```

### Configuration Parameters

Edit `train_dx_llava.sh` to customize:

```bash
# Model Configuration
MODEL_PATH="liuhaotian/llava-v1.5-7b"  # Base model
VISION_TOWER="hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup"

# Data Configuration
TRAIN_DATA_PATH="/path/to/train_data.json"
EVAL_DATA_PATH="/path/to/eval_data.json"
OUTPUT_DIR="/path/to/output"

# Training Hyperparameters
--learning_rate 2e-5
--per_device_train_batch_size 8
--num_train_epochs 1
--binary_loss_weight 10.0
```

## Data Preparation

### Dataset Format

Your training data should be in JSON format:

```json
{
  "id": "ddim/train/fake/example.png",
  "image": "ddim/train/fake/example.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nAnalyze this image for any signs of digital manipulation or artificial generation and determine if it is real or fake. Provide your reasoning first, then conclude with a letter answer."
    },
    {
      "from": "gpt",
      "value": "**Skin Texture Anomalies**: The skin exhibits unnatural smoothness and lacks realistic pores... ### Answer: fake"
    }
  ],
  "binary_label": 1
}
```

### Binary Classification Labels

For deepfake detection, use binary responses:
- `"real"`(0) for authentic images
- `"fake"`(1) for deepfakes

## Monitoring Training

### WandB Integration

The training automatically logs to Weights & Biases. Key metrics to monitor:

- **Training Loss**: Overall model loss
- **Binary Loss**: Classification-specific loss
- **LM Loss**: Language modeling loss
- **Learning Rate**: Cosine schedule progression

### Evaluation During Training

The model evaluates every 200 steps on the validation set. Check:

- **Eval Loss**: Validation performance
- **Binary Accuracy**: Classification accuracy
- **Perplexity**: Language generation quality

## Advanced Configuration

### Multi-GPU Training

Adjust GPU configuration:
```bash
# For 4 GPUs
--include localhost:0,1,2,3

# For 8 GPUs
--include localhost:0,1,2,3,4,5,6,7
```

### Memory Optimization

For limited VRAM:
```bash
--per_device_train_batch_size 4    # Reduce batch size
--gradient_accumulation_steps 2    # Maintain effective batch size
--gradient_checkpointing True      # Enable checkpointing
```

### Loss Weight Tuning

Balance language modeling and classification:
```bash
--lm_loss_weight 1.0        # Language modeling weight
--binary_loss_weight 10.0    # Binary classification weight
```