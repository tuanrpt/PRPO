#!/bin/bash

# Copyright (c) 2025 Tuan Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Full DX LLaVA Evaluation Pipeline
# Step 1: Create questions JSONL from dataset
# Step 2: Run DX LLaVA evaluation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
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

echo "🚀 FULL DX LLAVA EVALUATION PIPELINE"
printf '=%.0s' {1..60}
echo

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Step 1: Question creation parameters
INPUT_DATASET="../../df_data_reasoning/domain_splits/ddim/ddim_subset_1000.json"
QUESTION_OUTPUT="./question/df_question_ddim_1000.jsonl"
NUM_SAMPLES=1000
RANDOM_SEED=42

# Step 2: Evaluation parameters
MODEL_PATH="../checkpoints/dx-llava-binary-ddim"
IMAGE_FOLDER="../DF_R5_processed_images"
EVAL_OUTPUT="./answer/dx_llava_eval_ddim.jsonl"

# Evaluation settings
TEMPERATURE=0.2
MAX_NEW_TOKENS=3072
NUM_BEAMS=1
CONV_MODE="llava_v1"
PARSE_BINARY=true
CLEANUP_TEMP="--cleanup-temp-files"

# Multi-GPU settings
WORLD_SIZE=""
MASTER_PORT=""

# =============================================================================
# HELP FUNCTION
# =============================================================================

show_usage() {
    cat << EOF
Full DX LLaVA Evaluation Pipeline

Usage: $0 [OPTIONS]

Step 1 Options (Question Creation):
    --input-dataset PATH        Path to input dataset JSON (default: $INPUT_DATASET)
    --question-output PATH      Path to output questions JSONL (default: $QUESTION_OUTPUT)
    --num-samples INT           Number of samples to select (default: $NUM_SAMPLES)
    --random-seed INT           Random seed for sampling (default: $RANDOM_SEED)

Step 2 Options (Evaluation):
    --model-path PATH           Path to model checkpoint (default: $MODEL_PATH)
    --image-folder PATH         Path to images folder (default: $IMAGE_FOLDER)
    --eval-output PATH          Path to evaluation output file (default: $EVAL_OUTPUT)
    --conv-mode MODE            Conversation mode (default: $CONV_MODE)
    --temperature FLOAT         Temperature for generation (default: $TEMPERATURE)
    --num-beams INT             Number of beams (default: $NUM_BEAMS)
    --max-tokens INT            Maximum new tokens (default: $MAX_NEW_TOKENS)
    --world-size INT            Number of GPUs to use (default: auto-detect)
    --master-port INT           Master port for distributed processing (default: auto-select)

Control Options:
    --no-cleanup                Don't cleanup temporary files
    --no-binary-parsing         Don't parse binary classification outputs
    --force-single-gpu          Force single GPU mode
    --skip-questions            Skip question creation step (use existing questions)
    --skip-eval                 Skip evaluation step (only create questions)
    --dry-run                   Show what would be executed without running
    --help                      Show this help message

Examples:
    # Full pipeline with defaults
    $0

    # Custom dataset and output paths
    $0 --input-dataset /path/to/dataset.json --question-output /path/to/questions.jsonl

    # Only create questions (skip evaluation)
    $0 --skip-eval

    # Only run evaluation (skip question creation)
    $0 --skip-questions

    # Custom model and evaluation settings
    $0 --model-path /path/to/model --world-size 4 --temperature 0.0

EOF
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

SKIP_QUESTIONS=false
SKIP_EVAL=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        # Step 1 arguments
        --input-dataset)
            INPUT_DATASET="$2"
            shift 2
            ;;
        --question-output)
            QUESTION_OUTPUT="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --random-seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        # Step 2 arguments
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --image-folder)
            IMAGE_FOLDER="$2"
            shift 2
            ;;
        --eval-output)
            EVAL_OUTPUT="$2"
            shift 2
            ;;
        --conv-mode)
            CONV_MODE="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --num-beams)
            NUM_BEAMS="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --world-size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        # Control arguments
        --no-cleanup)
            CLEANUP_TEMP=""
            shift
            ;;
        --no-binary-parsing)
            PARSE_BINARY=false
            shift
            ;;
        --force-single-gpu)
            FORCE_SINGLE_GPU=true
            shift
            ;;
        --skip-questions)
            SKIP_QUESTIONS=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# =============================================================================
# VALIDATION
# =============================================================================

print_info "Validating inputs..."

# Check input dataset for step 1
if [[ "$SKIP_QUESTIONS" == "false" ]]; then
    if [[ ! -f "$INPUT_DATASET" ]]; then
        print_error "Input dataset does not exist: $INPUT_DATASET"
        exit 1
    fi
fi

# Check model and image folder for step 2
if [[ "$SKIP_EVAL" == "false" ]]; then
    if [[ ! -f "$MODEL_PATH/config.json" ]] && [[ ! -d "$MODEL_PATH" ]]; then
        print_error "Model path does not exist: $MODEL_PATH"
        exit 1
    fi

    if [[ ! -d "$IMAGE_FOLDER" ]]; then
        print_error "Image folder does not exist: $IMAGE_FOLDER"
        exit 1
    fi
fi

# Create output directories
if [[ "$SKIP_QUESTIONS" == "false" ]]; then
    QUESTION_DIR=$(dirname "$QUESTION_OUTPUT")
    if [[ ! -d "$QUESTION_DIR" ]]; then
        print_info "Creating question output directory: $QUESTION_DIR"
        mkdir -p "$QUESTION_DIR"
    fi
fi

if [[ "$SKIP_EVAL" == "false" ]]; then
    EVAL_DIR=$(dirname "$EVAL_OUTPUT")
    if [[ ! -d "$EVAL_DIR" ]]; then
        print_info "Creating evaluation output directory: $EVAL_DIR"
        mkdir -p "$EVAL_DIR"
    fi
fi

# =============================================================================
# STEP 1: CREATE QUESTIONS JSONL
# =============================================================================

if [[ "$SKIP_QUESTIONS" == "false" ]]; then
    print_info "🎯 STEP 1: Creating questions JSONL from dataset"
    echo

    # Create temporary Python script for question generation
    TEMP_SCRIPT="/tmp/create_questions_$$.py"
    cat > "$TEMP_SCRIPT" << 'EOF'
import json
import random
import sys
from collections import defaultdict

def create_data(data, output_jsonl_path, num_samples):
    """Randomly select samples and transform them to evaluation format."""
    print(f"\n--- Creating Random Subset ---")

    if len(data) < num_samples:
        print(f"Warning: Requested {num_samples} samples, but only {len(data)} are available.")
        print("Using all available samples instead.")
        num_samples = len(data)
    else:
        print(f"Randomly selecting {num_samples} samples...")
        data = random.sample(data, num_samples)

    processed_count = 0
    with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        for i, item in enumerate(data):
            try:
                image_path = item['image']
                question_text = item['conversations'][0]['value']
                gpt_response = item['conversations'][1]['value']

                # Extract label
                if '### Answer:' in gpt_response:
                    label = gpt_response.split('### Answer:')[-1].strip().lower()
                    if label not in ['real', 'fake']:
                        print(f"Warning: Skipping item with unexpected label: '{label}'")
                        continue
                else:
                    # Fallback for 'real' or 'fake' in the path
                    if 'real/' in image_path:
                        label = 'real'
                    elif 'fake/' in image_path:
                        label = 'fake'
                    else:
                        print(f"Warning: Could not determine label for image: {image_path}")
                        continue

                # Create evaluation record
                new_record = {
                    "question_id": processed_count,
                    "image": image_path,
                    "text": question_text,
                    "label": label
                }

                outfile.write(json.dumps(new_record) + '\n')
                processed_count += 1

            except (KeyError, IndexError) as e:
                print(f"Warning: Skipping malformed item #{i}. Error: {e}")
                continue

    print("-----------------------------------------")
    print(f"Successfully wrote {processed_count} samples to '{output_jsonl_path}'")
    return processed_count

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <input_file> <output_file> <num_samples> <random_seed>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    num_to_select = int(sys.argv[3])
    random_seed = int(sys.argv[4])

    # Set random seed for reproducible sampling
    random.seed(random_seed)
    print(f"Using random seed: {random_seed}")

    try:
        print(f"Reading data from '{input_file}'...")
        with open(input_file, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file '{input_file}' is not a valid JSON file.")
        sys.exit(1)

    created_samples = create_data(full_data, output_file, num_to_select)

EOF

    # Execute question creation
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "DRY RUN - Would execute: python $TEMP_SCRIPT '$INPUT_DATASET' '$QUESTION_OUTPUT' $NUM_SAMPLES $RANDOM_SEED"
    else
        if python "$TEMP_SCRIPT" "$INPUT_DATASET" "$QUESTION_OUTPUT" "$NUM_SAMPLES" "$RANDOM_SEED"; then
            print_success "Step 1 completed: Questions created at $QUESTION_OUTPUT"
        else
            print_error "Step 1 failed: Could not create questions"
            rm -f "$TEMP_SCRIPT"
            exit 1
        fi
    fi

    # Clean up temporary script
    rm -f "$TEMP_SCRIPT"
    echo

else
    print_info "🎯 STEP 1: SKIPPED - Using existing questions file"
    if [[ ! -f "$QUESTION_OUTPUT" ]]; then
        print_error "Questions file does not exist: $QUESTION_OUTPUT"
        print_info "Either create the file first or remove --skip-questions flag"
        exit 1
    fi
    echo
fi

# =============================================================================
# STEP 2: RUN DX LLAVA EVALUATION
# =============================================================================

if [[ "$SKIP_EVAL" == "false" ]]; then
    print_info "🎯 STEP 2: Running DX LLaVA evaluation"
    echo

    # Activate conda environment
    print_info "Activating conda environment 'llava'..."
    eval "$(conda shell.bash hook)"
    conda activate llava
    if [[ $? -eq 0 ]]; then
        print_success "Successfully activated conda environment 'llava'"
    else
        print_warning "Failed to activate conda environment 'llava', continuing anyway..."
    fi

    # Check if CUDA is available
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        print_error "CUDA is not available. This script requires CUDA."
        exit 1
    fi

    # Get GPU count if world size not specified
    if [[ -z "$WORLD_SIZE" ]]; then
        WORLD_SIZE=$(python -c "import torch; print(torch.cuda.device_count())")
        print_info "Auto-detected $WORLD_SIZE GPUs"
    else
        print_info "Using $WORLD_SIZE GPUs as specified"
    fi

    # Construct the evaluation command
    CMD="python eval_dx_llava_multi_gpu.py"
    CMD="$CMD --model-path '$MODEL_PATH'"
    CMD="$CMD --image-folder '$IMAGE_FOLDER'"
    CMD="$CMD --question-file '$QUESTION_OUTPUT'"
    CMD="$CMD --answers-file '$EVAL_OUTPUT'"
    CMD="$CMD --conv-mode '$CONV_MODE'"
    CMD="$CMD --temperature $TEMPERATURE"
    CMD="$CMD --num_beams $NUM_BEAMS"
    CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
    CMD="$CMD --world-size $WORLD_SIZE"

    if [[ -n "$MASTER_PORT" ]]; then
        CMD="$CMD --master-port $MASTER_PORT"
    fi
    if [[ -n "$CLEANUP_TEMP" ]]; then
        CMD="$CMD $CLEANUP_TEMP"
    fi
    if [ "$PARSE_BINARY" = true ]; then
        CMD="$CMD --parse-binary-output"
    fi
    if [ "$FORCE_SINGLE_GPU" = true ]; then
        CMD="$CMD --force-single-gpu"
    fi

    # Show what will be executed
    print_info "Evaluation command:"
    echo "$CMD"
    echo

    # Count questions and estimate time
    if [[ -f "$QUESTION_OUTPUT" ]]; then
        TOTAL_QUESTIONS=$(wc -l < "$QUESTION_OUTPUT")
        QUESTIONS_PER_GPU=$((TOTAL_QUESTIONS / WORLD_SIZE))
        ESTIMATED_TIME_MIN=$((QUESTIONS_PER_GPU * 3 / 60))
        print_info "Total questions to evaluate: $TOTAL_QUESTIONS"
        print_info "Estimated time: ~$ESTIMATED_TIME_MIN minutes (rough estimate)"
    fi

    # Execute evaluation
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "DRY RUN - Would execute evaluation command"
    else
        start_time=$(date +%s)
        print_info "Starting DX LLaVA evaluation..."

        if eval "$CMD"; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            duration_min=$((duration / 60))
            duration_sec=$((duration % 60))

            print_success "Step 2 completed: Evaluation finished successfully!"
            print_info "Evaluation time: ${duration_min}m ${duration_sec}s"
            print_info "Results saved to: $EVAL_OUTPUT"

            # Show evaluation summary
            if [[ -f "$EVAL_OUTPUT" ]]; then
                RESULT_COUNT=$(wc -l < "$EVAL_OUTPUT")
                print_info "Total results: $RESULT_COUNT"

                if command -v jq >/dev/null 2>&1; then
                    print_info "Evaluation Summary:"
                    DX_LLAVA_COUNT=$(grep -o '"preprocessing":"convnext_openclip"' "$EVAL_OUTPUT" | wc -l || echo "0")
                    FALLBACK_COUNT=$(grep -o '"preprocessing":"fallback"' "$EVAL_OUTPUT" | wc -l || echo "0")
                    echo "   - DX LLaVA preprocessing: $DX_LLAVA_COUNT"
                    echo "   - Fallback preprocessing: $FALLBACK_COUNT"

                    if [ "$PARSE_BINARY" = true ]; then
                        REAL_PRED=$(grep -o '"predict":"real"' "$EVAL_OUTPUT" | wc -l || echo "0")
                        FAKE_PRED=$(grep -o '"predict":"fake"' "$EVAL_OUTPUT" | wc -l || echo "0")
                        NONE_PRED=$(grep -o '"predict":"None"' "$EVAL_OUTPUT" | wc -l || echo "0")
                        echo "   - Predicted Real: $REAL_PRED"
                        echo "   - Predicted Fake: $FAKE_PRED"
                        echo "   - Failed to parse: $NONE_PRED"
                    fi
                fi
            fi
        else
            print_error "Step 2 failed: Evaluation error"
            exit 1
        fi
    fi

else
    print_info "🎯 STEP 2: SKIPPED - Evaluation not requested"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo
print_success "🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!"
echo

if [[ "$SKIP_QUESTIONS" == "false" ]]; then
    print_info "Step 1 Output: $QUESTION_OUTPUT"
fi

if [[ "$SKIP_EVAL" == "false" ]]; then
    print_info "Step 2 Output: $EVAL_OUTPUT"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    print_info "This was a dry run - no actual execution performed"
fi
