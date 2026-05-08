#!/bin/bash

# Copyright (c) 2025 Tuan Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Combined Embeddings Pipeline Runner
# This script runs the two-step embeddings pipeline:
# 1. Precompute embeddings from model and data
# 2. Create parquet dataset from embeddings

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
STEP="all"
BATCH_SIZE=8
CONV_MODE="llava_v1"
SUFFIX_PROMPT=""

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    --step STEP                Which step to run: precompute, parquet, or all (default: all)
    --model-path PATH          Path to trained LLaVA model (required for precompute/all)
    --data-path PATH           Path to training data JSON file (required for precompute/all)
    --image-folder PATH        Path to images folder (required for precompute/all)
    --embeddings-dir PATH      Directory containing embeddings (required for parquet)
    --domain NAME              Domain name (required for parquet, auto-detected for all)
    --output-dir PATH          Output directory (required)
    --batch-size SIZE          Batch size for processing (default: 8)
    --conv-mode MODE           Conversation mode (default: llava_v1)
    --suffix-prompt TEXT       Suffix prompt (default: empty)
    --help                     Show this help message

EXAMPLES:
    # Run complete pipeline
    $0 --step all \\
       --model-path /path/to/model \\
       --data-path /path/to/data.json \\
       --image-folder /path/to/images \\
       --output-dir /path/to/output \\
       --domain ddim

    # Only precompute embeddings
    $0 --step precompute \\
       --model-path /path/to/model \\
       --data-path /path/to/data.json \\
       --image-folder /path/to/images \\
       --output-dir /path/to/output

    # Only create parquet data
    $0 --step parquet \\
       --embeddings-dir /path/to/embeddings \\
       --domain ddim \\
       --output-dir /path/to/output

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            STEP="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --image-folder)
            IMAGE_FOLDER="$2"
            shift 2
            ;;
        --embeddings-dir)
            EMBEDDINGS_DIR="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --conv-mode)
            CONV_MODE="$2"
            shift 2
            ;;
        --suffix-prompt)
            SUFFIX_PROMPT="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ -z "$OUTPUT_DIR" ]]; then
    print_error "Output directory is required"
    usage
    exit 1
fi

if [[ "$STEP" == "precompute" || "$STEP" == "all" ]]; then
    if [[ -z "$MODEL_PATH" || -z "$DATA_PATH" || -z "$IMAGE_FOLDER" ]]; then
        print_error "For precompute step: --model-path, --data-path, and --image-folder are required"
        usage
        exit 1
    fi
fi

if [[ "$STEP" == "parquet" ]]; then
    if [[ -z "$EMBEDDINGS_DIR" || -z "$DOMAIN" ]]; then
        print_error "For parquet step: --embeddings-dir and --domain are required"
        usage
        exit 1
    fi
fi

# Validate step
if [[ "$STEP" != "precompute" && "$STEP" != "parquet" && "$STEP" != "all" ]]; then
    print_error "Invalid step: $STEP. Must be 'precompute', 'parquet', or 'all'"
    exit 1
fi

# Check if script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/combined_embeddings_pipeline.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
PYTHON_CMD="python $PYTHON_SCRIPT --step $STEP --output-dir $OUTPUT_DIR"

if [[ "$STEP" == "precompute" || "$STEP" == "all" ]]; then
    PYTHON_CMD="$PYTHON_CMD --model-path '$MODEL_PATH' --data-path '$DATA_PATH' --image-folder '$IMAGE_FOLDER'"
    PYTHON_CMD="$PYTHON_CMD --batch-size $BATCH_SIZE --conv-mode $CONV_MODE"

    if [[ -n "$DOMAIN" ]]; then
        PYTHON_CMD="$PYTHON_CMD --domain '$DOMAIN'"
    fi
fi

if [[ "$STEP" == "parquet" ]]; then
    PYTHON_CMD="$PYTHON_CMD --embeddings-dir '$EMBEDDINGS_DIR' --domain '$DOMAIN'"

    if [[ -n "$SUFFIX_PROMPT" ]]; then
        PYTHON_CMD="$PYTHON_CMD --suffix-prompt '$SUFFIX_PROMPT'"
    fi
fi

# Show configuration
print_status "Starting embeddings pipeline..."
echo "============================================"
echo "Step: $STEP"
echo "Output directory: $OUTPUT_DIR"

if [[ "$STEP" == "precompute" || "$STEP" == "all" ]]; then
    echo "Model path: $MODEL_PATH"
    echo "Data path: $DATA_PATH"
    echo "Image folder: $IMAGE_FOLDER"
    echo "Batch size: $BATCH_SIZE"
    echo "Conv mode: $CONV_MODE"
    [[ -n "$DOMAIN" ]] && echo "Domain: $DOMAIN"
fi

if [[ "$STEP" == "parquet" || "$STEP" == "all" ]]; then
    [[ -n "$EMBEDDINGS_DIR" ]] && echo "Embeddings directory: $EMBEDDINGS_DIR"
    [[ -n "$SUFFIX_PROMPT" ]] && echo "Suffix prompt: $SUFFIX_PROMPT"
fi

echo "============================================"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    print_status "Detected $GPU_COUNT GPU(s)"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    print_warning "nvidia-smi not found. GPU availability unknown."
fi

# Activate conda environment
if command -v conda &> /dev/null; then
    print_status "Activating llava conda environment..."
    # Source conda.sh to enable conda activate command
    eval "$(conda shell.bash hook)"
    if conda activate llava 2>/dev/null; then
        print_success "Successfully activated llava environment"
    else
        print_warning "Failed to activate llava environment. Continuing with current environment."
    fi
else
    print_warning "conda not found. Skipping environment activation."
fi

# Check CUDA availability
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    print_status "CUDA check passed"
else
    print_warning "CUDA check failed. Please ensure PyTorch with CUDA is installed."
fi

# Run the pipeline
print_status "Executing: $PYTHON_CMD"
echo ""

# Execute the command
if eval "$PYTHON_CMD"; then
    print_success "Pipeline completed successfully!"

    # Show output summary
    if [[ -d "$OUTPUT_DIR" ]]; then
        print_status "Output summary:"
        total_files=$(find "$OUTPUT_DIR" -type f | wc -l)
        total_size=$(du -sh "$OUTPUT_DIR" | cut -f1)
        echo "  📊 Total: $total_files files, $total_size"
    fi
else
    print_error "Pipeline failed!"
    exit 1
fi
