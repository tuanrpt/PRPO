#!/usr/bin/env python3

# Copyright (c) 2025 Tuan Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple evaluation script for PRPO saved models.
Uses the converted HuggingFace model format for easier loading.
"""

import json
import os
import sys
import time
import torch
import torch.multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from verl.utils.reward_score.deepfake_detection import extract_answer_deepfake, reward_func

def _deserialize_tensor(byte_data, tensor_info):
    """Deserialize bytes back to tensor using stored metadata."""
    if tensor_info is None or tensor_info.get("shape") is None:
        return byte_data

    shape = tensor_info["shape"]
    dtype_str = tensor_info["dtype"]

    if hasattr(shape, 'tolist'):
        shape = shape.tolist()
    elif not isinstance(shape, (list, tuple)):
        print(f"⚠️ Unexpected shape type: {type(shape)}, converting to list")
        shape = list(shape) if hasattr(shape, '__iter__') else [shape]

    if "bfloat16" in dtype_str:
        try:
            uint16_array = np.frombuffer(byte_data, dtype=np.uint16).reshape(shape)
            uint16_array = uint16_array.copy()
            tensor = torch.from_numpy(uint16_array).view(torch.bfloat16)
            return tensor
        except Exception as e:
            print(f"⚠️ bfloat16 deserialization failed: {e}, falling back to float32")
            try:
                float_array = np.frombuffer(byte_data, dtype=np.float32).reshape(shape)
                float_array = float_array.copy()
                tensor = torch.from_numpy(float_array)
                return tensor
            except Exception as e2:
                print(f"❌ Float32 fallback also failed: {e2}, returning dummy tensor")
                return torch.zeros(shape, dtype=torch.float32)

    elif "float16" in dtype_str:
        numpy_dtype = np.float16
    elif "float32" in dtype_str:
        numpy_dtype = np.float32
    elif "int64" in dtype_str:
        numpy_dtype = np.int64
    elif "int32" in dtype_str:
        numpy_dtype = np.int32
    else:
        numpy_dtype = np.float32

    try:
        numpy_array = np.frombuffer(byte_data, dtype=numpy_dtype).reshape(shape)
        numpy_array = numpy_array.copy()
        tensor = torch.from_numpy(numpy_array)
        return tensor

    except Exception as e:
        print(f"⚠️ Tensor deserialization failed for dtype {dtype_str}: {e}")
        print(f"   Data shape expected: {shape}, dtype: {numpy_dtype}")
        print(f"   Byte data length: {len(byte_data) if isinstance(byte_data, bytes) else 'not bytes'}")

        if "int" in dtype_str:
            return torch.zeros(shape, dtype=torch.int64)
        else:
            return torch.zeros(shape, dtype=torch.float32)

def get_available_gpus():
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        return []

    gpu_count = torch.cuda.device_count()
    gpu_info = []

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        gpu_info.append({
            'id': i,
            'name': props.name,
            'memory_gb': memory_gb
        })

    print(f"Detected {gpu_count} GPUs:")
    for gpu in gpu_info:
        print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")

    return [gpu['id'] for gpu in gpu_info]

def split_dataframe_for_gpus(df: pd.DataFrame, num_gpus: int) -> List[pd.DataFrame]:
    """Split dataframe into chunks for multi-GPU processing."""
    if num_gpus <= 1:
        return [df]

    chunk_size = len(df) // num_gpus
    chunks = []

    for i in range(num_gpus):
        start_idx = i * chunk_size
        if i == num_gpus - 1:  # Last chunk gets remaining samples
            end_idx = len(df)
        else:
            end_idx = (i + 1) * chunk_size

        chunk = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        chunks.append(chunk)

    chunks = [chunk for chunk in chunks if len(chunk) > 0]

    print(f"Split {len(df)} samples across {len(chunks)} GPUs:")
    for i, chunk in enumerate(chunks):
        print(f"  GPU {i}: {len(chunk)} samples")

    return chunks

def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint in the directory."""
    if os.path.isfile(os.path.join(checkpoint_dir, 'latest_checkpointed_iteration.txt')):
        with open(os.path.join(checkpoint_dir, 'latest_checkpointed_iteration.txt'), 'r') as f:
            step = f.read().strip()
        return os.path.join(checkpoint_dir, f'global_step_{step}')
    else:
        # Find the latest step directory
        step_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith('global_step_')]
        if not step_dirs:
            raise ValueError(f"No checkpoint directories found in {checkpoint_dir}")

        latest_step = max(step_dirs, key=lambda x: int(x.split('_')[-1]))
        return os.path.join(checkpoint_dir, latest_step)

def check_for_model_files(hf_model_path: str) -> bool:
    """Check if the HuggingFace model directory contains actual model files."""
    model_files = [
        "pytorch_model.bin", "model.safetensors", "tf_model.h5",
        "model.ckpt.index", "flax_model.msgpack"
    ]

    for model_file in model_files:
        if os.path.exists(os.path.join(hf_model_path, model_file)):
            return True

    bin_files = [f for f in os.listdir(hf_model_path) if f.startswith("pytorch_model") and f.endswith(".bin")]
    safetensors_files = [f for f in os.listdir(hf_model_path) if f.startswith("model") and f.endswith(".safetensors")]

    return len(bin_files) > 0 or len(safetensors_files) > 0

def load_model_from_checkpoint(checkpoint_path: str, gpu_id: int = None, multi_gpu: bool = False):
    """Load model from PRPO checkpoint using the HuggingFace format."""
    print(f"Loading model from PRPO checkpoint: {checkpoint_path}")

    hf_model_path = os.path.join(checkpoint_path, "actor", "huggingface")

    if os.path.exists(hf_model_path) and check_for_model_files(hf_model_path):
        print(f"Found complete HuggingFace model at: {hf_model_path}")
    elif os.path.exists(hf_model_path):
        print(f"Warning: HuggingFace directory exists but no model weights found at {hf_model_path}")
        print("The checkpoint appears to be in distributed/FSDP format.")
        print("Using fallback base model (not PRPO-trained weights)")
    else:
        print(f"Warning: No HuggingFace model found at {hf_model_path}")
        print("Using fallback base model (not PRPO-trained weights)")

    print(f"Loading model from: {hf_model_path}")
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)

        if multi_gpu and gpu_id is None:
            device_map = "auto"
        elif gpu_id is not None:
            device_map = {f"": gpu_id}
        else:
            device_map = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map=device_map
        )

        model.eval()
        print(f"✅ Successfully loaded model from {hf_model_path}")

        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def generate_responses_with_embeddings(model, tokenizer, test_df: pd.DataFrame, print_responses: bool = True, output_dir: str = None) -> pd.DataFrame:
    """Generate responses using embeddings with the loaded model."""
    print(f"Generating responses for {len(test_df)} samples using embeddings...")

    responses = []
    generation_times = []
    response_lengths = []

    log_file_path = None
    log_file = None
    if output_dir:
        log_file_path = os.path.join(output_dir, "detailed_generation_log.txt")
        log_file = open(log_file_path, 'w', encoding='utf-8')
        log_file.write("PRPO EVALUATION - DETAILED GENERATION LOG\n")
        log_file.write("="*80 + "\n\n")

    total_start_time = time.time()

    for idx, row in test_df.iterrows():
        if idx % 10 == 0:
            print(f"Processing sample {idx+1}/{len(test_df)}")

        sample_start_time = time.time()
        response = "ERROR: Failed to generate response"

        try:
            inputs_embeds_data = row['inputs_embeds']
            attention_mask_data = row['attention_mask']
            extra_info = row.get('extra_info', {})
            tensor_info = extra_info.get('tensor_info', {})

            if isinstance(inputs_embeds_data, bytes):
                inputs_embeds_info = tensor_info.get('inputs_embeds', {})
                attention_mask_info = tensor_info.get('attention_mask', {})

                inputs_embeds = _deserialize_tensor(inputs_embeds_data, inputs_embeds_info)
                attention_mask = _deserialize_tensor(attention_mask_data, attention_mask_info)
            else:
                inputs_embeds = inputs_embeds_data
                attention_mask = attention_mask_data

            model_dtype = next(model.parameters()).dtype
            inputs_embeds = inputs_embeds.to(device=model.device, dtype=model_dtype)
            attention_mask = attention_mask.to(device=model.device)

            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            generation_start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=3072,
                    temperature=0.6,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            generation_end_time = time.time()
            generation_time = generation_end_time - generation_start_time

            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response.strip()

            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            response_length = len(response_tokens)
            response_lengths.append(response_length)

            print(f"     ✅ Generated in {generation_time:.2f}s (Length: {response_length} tokens)")

            reward_model = row.get('reward_model', {})
            if isinstance(reward_model, dict) and 'ground_truth' in reward_model:
                ground_truth = reward_model['ground_truth']
            else:
                ground_truth = 'N/A'

            extra_info = row.get('extra_info', {})
            image_path = extra_info.get('image_path', 'N/A')

            if log_file:
                log_file.write(f"SAMPLE {idx + 1}\n")
                log_file.write(f"Generated in {generation_time:.2f}s (Length: {response_length} tokens)\n")
                log_file.write(f"Ground Truth: {ground_truth}\n")
                log_file.write(f"Image Path: {image_path}\n")
                log_file.write(f"Response: {response}\n")
                log_file.write("="*80 + "\n\n")
                log_file.flush()

            if print_responses:
                print(f"     🎯 Ground Truth: {ground_truth}")
                print(f"     🖼️  Image Path: {image_path}")
                print(f"     📝 Response: {response}")
                print("     " + "="*60)

        except Exception as e:
            print(f"     ❌ Error processing sample {idx}: {str(e)}")
            response = f"ERROR: {str(e)}"
            generation_time = time.time() - sample_start_time
            response_lengths.append(len(response))

            if log_file:
                log_file.write(f"SAMPLE {idx + 1}\n")
                log_file.write(f"Error processing sample in {generation_time:.2f}s\n")
                log_file.write(f"Ground Truth: N/A\n")
                log_file.write(f"Image Path: N/A\n")
                log_file.write(f"Response: {response}\n")
                log_file.write("="*80 + "\n\n")
                log_file.flush()

        responses.append(response)
        generation_times.append(generation_time)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    if log_file:
        log_file.close()
        print(f"Detailed generation log saved to: {log_file_path}")

    print_timing_and_length_stats(generation_times, total_time, response_lengths)

    if output_dir:
        save_response_length_stats(response_lengths, output_dir)

    test_df = test_df.copy()
    test_df['generated_response'] = responses
    test_df['generation_time'] = generation_times

    return test_df

def print_timing_and_length_stats(generation_times: List[float], total_time: float, response_lengths: List[int]):
    """Print generation timing and response length statistics."""
    avg_time = np.mean(generation_times)
    min_time = np.min(generation_times)
    max_time = np.max(generation_times)
    std_time = np.std(generation_times)

    avg_length = np.mean(response_lengths)
    min_length = np.min(response_lengths)
    max_length = np.max(response_lengths)
    std_length = np.std(response_lengths)

    print(f"\n{'='*70}")
    print(f"GENERATION STATISTICS")
    print(f"{'='*70}")
    print(f"Total samples: {len(generation_times)}")
    print()

    print("TIMING STATISTICS:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per sample: {avg_time:.2f}s")
    print(f"  Min time: {min_time:.2f}s")
    print(f"  Max time: {max_time:.2f}s")
    print(f"  Std deviation: {std_time:.2f}s")
    print(f"  Throughput: {len(generation_times)/total_time:.2f} samples/sec")
    print()

    print("RESPONSE LENGTH STATISTICS:")
    print(f"  Average length: {avg_length:.1f} tokens")
    print(f"  Min length: {min_length} tokens")
    print(f"  Max length: {max_length} tokens")
    print(f"  Std deviation: {std_length:.1f} tokens")
    print(f"{'='*70}")

def save_response_length_stats(response_lengths: List[int], output_dir: str):
    """Save response length statistics to JSON and CSV files."""
    avg_length = float(np.mean(response_lengths))
    min_length = int(np.min(response_lengths))
    max_length = int(np.max(response_lengths))
    std_length = float(np.std(response_lengths))
    median_length = float(np.median(response_lengths))

    p25 = float(np.percentile(response_lengths, 25))
    p75 = float(np.percentile(response_lengths, 75))
    p90 = float(np.percentile(response_lengths, 90))
    p95 = float(np.percentile(response_lengths, 95))
    p99 = float(np.percentile(response_lengths, 99))
    stats = {
        'total_samples': len(response_lengths),
        'mean_tokens': avg_length,
        'min_tokens': min_length,
        'max_tokens': max_length,
        'std_tokens': std_length,
        'median_tokens': median_length,
        'percentiles': {
            'p25': p25,
            'p75': p75,
            'p90': p90,
            'p95': p95,
            'p99': p99
        },
        'response_lengths': response_lengths
    }

    stats_file = os.path.join(output_dir, "response_length_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    summary_stats = {
        'metric': ['total_samples', 'mean_tokens', 'min_tokens', 'max_tokens', 'std_tokens', 'median_tokens', 'p25', 'p75', 'p90', 'p95', 'p99'],
        'value': [len(response_lengths), avg_length, min_length, max_length, std_length, median_length, p25, p75, p90, p95, p99]
    }

    import pandas as pd
    summary_df = pd.DataFrame(summary_stats)
    summary_file = os.path.join(output_dir, "response_length_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    lengths_df = pd.DataFrame({
        'sample_index': range(1, len(response_lengths) + 1),
        'response_length_tokens': response_lengths
    })
    lengths_file = os.path.join(output_dir, "individual_response_lengths.csv")
    lengths_df.to_csv(lengths_file, index=False)

    print(f"Response length statistics saved to:")
    print(f"  - {stats_file}")
    print(f"  - {summary_file}")
    print(f"  - {lengths_file}")

def evaluate_with_prpo_reward(df_with_responses: pd.DataFrame) -> tuple:
    """Evaluate responses using PRPO reward function."""
    print("Evaluating responses with PRPO reward function...")

    results = []
    scores = []
    predictions = []
    ground_truths = []

    for idx, row in df_with_responses.iterrows():
        if idx % 50 == 0:
            print(f"Evaluating sample {idx+1}/{len(df_with_responses)}")

        response = str(row['generated_response'])
        data_source = row.get('data_source', 'ddim')

        reward_model = row.get('reward_model', {})
        if isinstance(reward_model, dict) and 'ground_truth' in reward_model:
            ground_truth = reward_model['ground_truth']
        else:
            ground_truth = 'unknown'

        extra_info = row.get('extra_info', {})

        try:
            score = reward_func(data_source, response, ground_truth, extra_info)
            if isinstance(score, dict):
                score_val = score.get('score', 0.0)
            else:
                score_val = float(score)
        except Exception as e:
            print(f"Error computing score for sample {idx}: {e}")
            score_val = 0.0

        prediction = extract_answer_deepfake(response)

        result = {
            'index': idx,
            'response': response,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'score': score_val,
            'data_source': data_source
        }

        results.append(result)
        scores.append(score_val)

        if ground_truth in ['real', 'fake'] and prediction in ['real', 'fake']:
            ground_truths.append(ground_truth)
            predictions.append(prediction)

    return results, predictions, ground_truths, scores

def compute_comprehensive_metrics(results: list, predictions: list, ground_truths: list, scores: list):
    """Compute comprehensive evaluation metrics."""
    print(f"\nComputing metrics...")
    print(f"Total samples: {len(results)}")
    print(f"Valid predictions: {len(predictions)}")

    prpo_accuracy = np.mean(scores)
    if len(predictions) > 0:
        sklearn_accuracy = accuracy_score(ground_truths, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truths, predictions, labels=['real', 'fake'], average=None
        )

        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        cm = confusion_matrix(ground_truths, predictions, labels=['real', 'fake'])

        class_report = classification_report(
            ground_truths, predictions,
            labels=['real', 'fake'],
            target_names=['Real', 'Fake'],
            output_dict=True
        )
    else:
        sklearn_accuracy = 0.0
        precision = recall = f1 = support = [0.0, 0.0]
        macro_precision = macro_recall = macro_f1 = 0.0
        cm = np.zeros((2, 2))
        class_report = {}

    metrics = {
        'prpo_accuracy': prpo_accuracy,
        'sklearn_accuracy': sklearn_accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class_metrics': {
            'real': {
                'precision': float(precision[0]) if len(precision) > 0 else 0.0,
                'recall': float(recall[0]) if len(recall) > 0 else 0.0,
                'f1': float(f1[0]) if len(f1) > 0 else 0.0,
                'support': int(support[0]) if len(support) > 0 else 0
            },
            'fake': {
                'precision': float(precision[1]) if len(precision) > 1 else 0.0,
                'recall': float(recall[1]) if len(recall) > 1 else 0.0,
                'f1': float(f1[1]) if len(f1) > 1 else 0.0,
                'support': int(support[1]) if len(support) > 1 else 0
            }
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'total_samples': len(results),
        'valid_predictions': len(predictions),
        'invalid_predictions': len(results) - len(predictions)
    }

    return metrics

def print_evaluation_results(metrics: dict):
    """Print evaluation results."""
    print("\n" + "="*80)
    print("PRPO SAVED MODEL EVALUATION RESULTS")
    print("="*80)

    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Valid Predictions: {metrics['valid_predictions']}")
    print(f"Invalid Predictions: {metrics['invalid_predictions']}")
    print()

    print("ACCURACY METRICS:")
    print(f"PRPO Accuracy (Mean Score): {metrics['prpo_accuracy']:.4f} ({metrics['prpo_accuracy']*100:.2f}%)")
    print(f"Classification Accuracy: {metrics['sklearn_accuracy']:.4f} ({metrics['sklearn_accuracy']*100:.2f}%)")
    print()

    print("CLASSIFICATION METRICS:")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    print()

    print("PER-CLASS PERFORMANCE:")
    for class_name, class_metrics in metrics['per_class_metrics'].items():
        print(f"  {class_name.upper()}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall: {class_metrics['recall']:.4f}")
        print(f"    F1-Score: {class_metrics['f1']:.4f}")
        print(f"    Support: {class_metrics['support']}")
        print()

    if metrics['valid_predictions'] > 0:
        print("CONFUSION MATRIX:")
        cm = metrics['confusion_matrix']
        print("           Predicted")
        print("         Real  Fake")
        print(f"Real   {cm[0][0]:6.0f}  {cm[0][1]:4.0f}")
        print(f"Fake   {cm[1][0]:6.0f}  {cm[1][1]:4.0f}")

    print("="*80)

def generate_responses_on_gpu(args_tuple):
    """Worker function for multi-GPU generation."""
    gpu_id, checkpoint_path, test_df_chunk, print_responses, worker_id = args_tuple

    try:
        torch.cuda.set_device(gpu_id)

        print(f"GPU {gpu_id} (Worker {worker_id}): Processing {len(test_df_chunk)} samples")

        model, tokenizer = load_model_from_checkpoint(checkpoint_path, gpu_id=gpu_id)

        df_with_responses = generate_responses_with_embeddings(
            model, tokenizer, test_df_chunk, print_responses
        )

        print(f"GPU {gpu_id} (Worker {worker_id}): Completed {len(df_with_responses)} samples")

        del model
        torch.cuda.empty_cache()

        return df_with_responses, gpu_id

    except Exception as e:
        print(f"GPU {gpu_id} (Worker {worker_id}): Error - {e}")
        import traceback
        traceback.print_exc()
        return None, gpu_id

def run_multi_gpu_evaluation(checkpoint_path: str, test_df_sampled: pd.DataFrame, print_responses: bool) -> pd.DataFrame:
    """Run evaluation across multiple GPUs."""
    available_gpus = get_available_gpus()

    if len(available_gpus) <= 1:
        print("Single GPU detected, falling back to single GPU evaluation")
        model, tokenizer = load_model_from_checkpoint(checkpoint_path, multi_gpu=True)
        return generate_responses_with_embeddings(model, tokenizer, test_df_sampled, print_responses)

    print(f"Multi-GPU evaluation using {len(available_gpus)} GPUs")

    data_chunks = split_dataframe_for_gpus(test_df_sampled, len(available_gpus))

    worker_args = []
    for i, (gpu_id, chunk) in enumerate(zip(available_gpus, data_chunks)):
        worker_args.append((gpu_id, checkpoint_path, chunk, print_responses, i))

    print("Starting multi-GPU generation...")
    all_results = []

    mp.set_start_method('spawn', force=True)

    with ProcessPoolExecutor(max_workers=len(available_gpus)) as executor:
        futures = [executor.submit(generate_responses_on_gpu, args) for args in worker_args]

        for future in as_completed(futures):
            result, gpu_id = future.result()
            if result is not None:
                all_results.append(result)
                print(f"✅ GPU {gpu_id} completed successfully")
            else:
                print(f"❌ GPU {gpu_id} failed")

    if not all_results:
        raise RuntimeError("All GPU workers failed")

    print(f"Combining results from {len(all_results)} GPUs...")
    combined_df = pd.concat(all_results, ignore_index=True)

    if 'original_index' in combined_df.columns:
        combined_df = combined_df.sort_values('original_index').reset_index(drop=True)

    print(f"Combined {len(combined_df)} total responses from all GPUs")

    return combined_df

def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate saved PRPO model (simple version)")
    parser.add_argument("--checkpoint_dir",
                       default=None,
                       help="Path to PRPO checkpoint directory")
    parser.add_argument("--test_file",
                       default=None,
                       help="Path to test parquet file")
    parser.add_argument("--sample_size", type=int, default=-1,
                       help="Number of samples to evaluate (default: 50, -1 for all)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for sampling")
    parser.add_argument("--print_responses", action="store_true",
                       help="Print full responses during generation")
    parser.add_argument("--multi_gpu", action="store_true",
                       help="Enable multi-GPU parallel generation (auto-detects available GPUs)")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, f"prpo_simple_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("PRPO SAVED MODEL EVALUATION (SIMPLE)")
    print("="*80)
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
    print(f"Test Data: {args.test_file}")
    print(f"Output: {output_dir}")
    print()

    try:
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        print(f"Using checkpoint: {latest_checkpoint}")

        print("Loading test data...")
        test_df = pd.read_parquet(args.test_file)
        print(f"Loaded {len(test_df)} test samples")

        if args.sample_size == -1 or len(test_df) <= args.sample_size:
            print(f"Using all {len(test_df)} samples")
            test_df_sampled = test_df
        else:
            print(f"Randomly sampling {args.sample_size} samples...")
            test_df_sampled = test_df.sample(n=args.sample_size, random_state=args.random_seed).reset_index(drop=True)
            print(f"Selected {len(test_df_sampled)} samples")

        if args.multi_gpu:
            print("🚀 Multi-GPU evaluation enabled")
            df_with_responses = run_multi_gpu_evaluation(
                latest_checkpoint, test_df_sampled, args.print_responses
            )
        else:
            print("🚀 Single-GPU evaluation")
            model, tokenizer = load_model_from_checkpoint(latest_checkpoint)
            df_with_responses = generate_responses_with_embeddings(
                model, tokenizer, test_df_sampled, args.print_responses, output_dir
            )

        responses_file = os.path.join(output_dir, "prpo_simple_responses.parquet")
        df_with_responses.to_parquet(responses_file, index=False)
        print(f"Saved responses to: {responses_file}")

        results, predictions, ground_truths, scores = evaluate_with_prpo_reward(df_with_responses)

        results_file = os.path.join(output_dir, "prpo_simple_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        metrics = compute_comprehensive_metrics(results, predictions, ground_truths, scores)

        metrics_file = os.path.join(output_dir, "prpo_simple_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print_evaluation_results(metrics)

        print(f"\nEvaluation complete! Results saved in: {output_dir}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
