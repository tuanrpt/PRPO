#!/bin/bash

# Copyright (c) 2025 Tuan Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Convert FSDP checkpoint to HuggingFace format and run multi-GPU evaluation
set -e

echo "🔧 Converting FSDP checkpoint to HuggingFace format..."

DOMAIN="ddim"
CHECKPOINT_DIR="../checkpoints/dx_llava_prpo_${DOMAIN}"
FSDP_CHECKPOINT_PATH="${CHECKPOINT_DIR}/global_step_93/actor"
HF_OUTPUT_PATH="${CHECKPOINT_DIR}/global_step_93/actor/huggingface"
TEST_FILE="../data/${DOMAIN}/test.parquet"

# Step 1: Convert FSDP to HuggingFace format
echo "📦 Converting FSDP checkpoint..."
echo "   From: ${FSDP_CHECKPOINT_PATH}"
echo "   To: ${HF_OUTPUT_PATH}"

# Unset problematic environment variables
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
unset HIP_VISIBLE_DEVICES 2>/dev/null || true

# Run conversion using official PRPO model merger
cd ..
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${FSDP_CHECKPOINT_PATH}" \
    --target_dir "${HF_OUTPUT_PATH}"

# Return to original directory
cd eval

if [ $? -ne 0 ]; then
    echo "❌ FSDP conversion failed!"
    exit 1
fi

echo "✅ FSDP conversion completed!"

# Step 2: Run multi-GPU evaluation
echo ""
echo "🚀 Starting multi-GPU evaluation..."

python evaluate_prpo_simple.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --multi_gpu \
    --sample_size -1 \
    --print_responses \
    --test_file "${TEST_FILE}"

echo "✅ Evaluation completed!"
