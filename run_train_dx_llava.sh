#!/bin/bash

# DX-LLaVA Training Script
# Fine-tuning LLaVA with ConvNeXt vision encoder for deepfake detection
#
# Usage: ./train_dx_llava.sh
#
# Requirements:
# - DeepSpeed installed
# - At least 2 H100 GPUs available (localhost:0,1)
# - Appropriate data paths configured below

# Configuration - Modify these paths according to your setup
MODEL_PATH="liuhaotian/llava-v1.5-7b"
TRAIN_DATA_PATH="domain_splits/ddim/train_91977.json"  # Relative to github_version directory
OUTPUT_DIR="../checkpoints/dx-llava-binary-ddim"                 # Relative to github_version directory
IMAGE_FOLDER="../DF_R5_processed_images"
VISION_TOWER="hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup"

echo "Starting DX-LLaVA training..."
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"

PYTHONPATH=. deepspeed --master_port 27000 --include localhost:0,1 llava/train/train_mem.py \
    --deepspeed ../scripts/zero3.json \
    --model_name_or_path "$MODEL_PATH" \
    --version v1 \
    --data_path "$TRAIN_DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 999999 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --use_image_augmentation False \
    --use_binary_labels True \
    --lm_loss_weight 1.0 \
    --binary_loss_weight 10.0 \
    --disable_tqdm True \

echo "Training completed!"
