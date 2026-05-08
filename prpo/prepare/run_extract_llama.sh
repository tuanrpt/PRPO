#!/bin/bash

# Copyright (c) 2025 Tuan Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Extract LLaMA weights from LLaVA model
# This script runs the extract_llama_weights.py with configurable paths

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

# Default configurations
LLAVA_MODEL_PATH="../checkpoints/dx-llava-binary-ddim"
OUTPUT_PATH="../checkpoints/llama-extracted-from-dx-llava-ddim"
PYTHON_SCRIPT="./extract_llama_weights.py"

# Help function
show_usage() {
    cat << EOF
Extract LLaMA weights from LLaVA model

Usage: $0 [OPTIONS]

Options:
    --llava-model PATH      Path to LLaVA model (default: $LLAVA_MODEL_PATH)
    --output PATH           Output path for extracted LLaMA model (default: $OUTPUT_PATH)
    --help                  Show this help message

Examples:
    # Extract with default paths
    $0

    # Extract with custom paths
    $0 --llava-model /path/to/llava/model --output /path/to/output

    # Extract different model variants
    $0 --llava-model ../checkpoints/dx-llava-binary-SiT --output ../checkpoints/llama-from-SiT
    $0 --llava-model ../checkpoints/dx-llava-binary-pixart --output ../checkpoints/llama-from-pixart

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --llava-model)
            LLAVA_MODEL_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
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

# Validation
print_info "Validating inputs..."

if [[ ! -d "$LLAVA_MODEL_PATH" ]]; then
    print_error "LLaVA model path does not exist: $LLAVA_MODEL_PATH"
    exit 1
fi

if [[ ! -f "$LLAVA_MODEL_PATH/config.json" ]]; then
    print_error "LLaVA model config not found: $LLAVA_MODEL_PATH/config.json"
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Create output directory
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
if [[ ! -d "$OUTPUT_DIR" ]]; then
    print_info "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

echo "🚀 EXTRACT LLAMA FROM LLAVA"
printf '=%.0s' {1..60}
echo

print_info "Configuration:"
echo "  - LLaVA Model: $LLAVA_MODEL_PATH"
echo "  - Output Path: $OUTPUT_PATH"
echo "  - Python Script: $PYTHON_SCRIPT"
echo

# Create a temporary Python script with the correct paths
TEMP_SCRIPT="/tmp/extract_llama_$$.py"
print_info "Creating temporary extraction script..."

# Copy everything except the main function, then add our custom main
sed -n '1,215p' "$PYTHON_SCRIPT" > "$TEMP_SCRIPT"

# Add our custom main function with the provided paths
cat >> "$TEMP_SCRIPT" << EOF

def main():
    llava_model_path = "$LLAVA_MODEL_PATH"
    output_path = "$OUTPUT_PATH"

    print("🚀 Starting optimized LLaMA weight extraction")
    print("=" * 60)

    # Extract weights
    print("1️⃣ Extracting weights...")
    extracted_weights = extract_llama_weights_optimized(llava_model_path, output_path)

    # Create config
    print("\\n2️⃣ Creating LLaMA config...")
    llava_config_path = os.path.join(llava_model_path, "config.json")
    create_llama_config(llava_config_path, output_path)

    # Copy tokenizer
    print("\\n3️⃣ Copying tokenizer files...")
    copy_tokenizer_files(llava_model_path, output_path)

    # Create generation config
    print("\\n4️⃣ Creating generation config...")
    create_generation_config(output_path)

    print("\\n" + "=" * 60)
    print("✅ LLaMA extraction completed!")
    print(f"📁 Extracted model: {output_path}")

if __name__ == "__main__":
    main()
EOF

print_info "Starting LLaMA weight extraction..."
start_time=$(date +%s)

# Activate conda environment if available
if command -v conda >/dev/null 2>&1; then
    print_info "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate llava || print_warning "Failed to activate llava environment, continuing anyway..."
fi

# Run the extraction
if python "$TEMP_SCRIPT"; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    duration_min=$((duration / 60))
    duration_sec=$((duration % 60))

    print_success "LLaMA extraction completed successfully!"
    print_info "Extraction time: ${duration_min}m ${duration_sec}s"
    print_info "Extracted model saved to: $OUTPUT_PATH"

    # Show extracted model info
    if [[ -f "$OUTPUT_PATH/config.json" ]]; then
        print_info "Extracted model configuration:"
        echo "  - Config: $OUTPUT_PATH/config.json"
        echo "  - Weights: $OUTPUT_PATH/model.safetensors"
        echo "  - Tokenizer: $OUTPUT_PATH/tokenizer.model"
        echo "  - Generation config: $OUTPUT_PATH/generation_config.json"

        # Check file sizes
        if command -v du >/dev/null 2>&1; then
            MODEL_SIZE=$(du -sh "$OUTPUT_PATH" 2>/dev/null | cut -f1 || echo "unknown")
            echo "  - Total size: $MODEL_SIZE"
        fi
    fi
else
    print_error "LLaMA extraction failed!"
    rm -f "$TEMP_SCRIPT"
    exit 1
fi

# Clean up
rm -f "$TEMP_SCRIPT"

echo
print_success "🎉 EXTRACTION COMPLETED SUCCESSFULLY!"
echo
print_info "You can now use the extracted LLaMA model at: $OUTPUT_PATH"
