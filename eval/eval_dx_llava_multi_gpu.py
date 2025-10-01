#!/usr/bin/env python3

# Copyright (c) 2025 Tuan Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Enhanced Multi-GPU LLaVA Evaluation Script with ConvNeXT CLIP Integration
Ensures proper preprocessing and model loading for DX-LLaVA models.
"""

import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import json
from tqdm import tqdm
import shortuuid
import time
from PIL import Image
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """Get the k-th chunk from a list split into n chunks"""
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


def setup_distributed(rank, world_size, master_port):
    """Initialize distributed processing"""
    from datetime import timedelta

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)

    print(f"[GPU {rank}] Setting up distributed processing on port {master_port}")

    try:
        timeout = timedelta(seconds=30)
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=timeout
        )
        torch.cuda.set_device(rank)
        print(f"[GPU {rank}] Distributed setup successful")
    except Exception as e:
        print(f"[GPU {rank}] Failed to setup distributed processing: {e}")
        print(f"[GPU {rank}] This usually indicates a distributed processing issue")
        print(f"[GPU {rank}] Try running with --force-single-gpu for debugging")
        raise


def cleanup_distributed():
    """Clean up distributed processing"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Warning: Failed to cleanup distributed process group: {e}")


def expand2square(pil_img, background_color):
    """Expand image to square by padding - matches training preprocessing"""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_convnext_image(image, vision_tower, model_config):
    """
    Process image using ConvNeXT preprocessing to match training pipeline exactly

    Args:
        image: PIL Image
        vision_tower: Vision tower with ConvNeXT CLIP
        model_config: Model configuration

    Returns:
        torch.Tensor: Processed image tensor
    """
    image_processor = vision_tower.image_processor

    if hasattr(image_processor, 'openclip_preprocess') and image_processor.openclip_preprocess is not None:
        processed_image = image_processor.openclip_preprocess(image)
        return processed_image

    elif hasattr(vision_tower, 'openclip_preprocess') and vision_tower.openclip_preprocess is not None:
        processed_image = vision_tower.openclip_preprocess(image)
        return processed_image

    else:
        try:
            import open_clip
            _, _, openclip_preprocess = open_clip.create_model_and_transforms(
                "convnext_large_d_320",
                pretrained="laion2b_s29b_b131k_ft_soup",
                device="cpu"
            )
            processed_image = openclip_preprocess(image)
            return processed_image
        except Exception as e:
            print(f"[WARNING] Failed to create OpenCLIP preprocessing: {e}")

    image_aspect_ratio = getattr(model_config, "image_aspect_ratio", "pad")

    if image_aspect_ratio == 'pad':
        background_color = tuple(int(x*255) for x in image_processor.image_mean)
        image = expand2square(image, background_color)

    processed_image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    return processed_image


def fix_convnext_config(model):
    """
    Fix configuration mismatch between ConvNeXT vision tower and model config.

    The trained model uses ConvNeXT (1536 hidden size) but the config shows
    standard CLIP ViT (1024 hidden size). This causes tensor dimension errors.
    """
    vision_tower = model.get_vision_tower()
    if vision_tower and vision_tower.use_openclip:
        actual_hidden_size = vision_tower.hidden_size
        config_hidden_size = model.config.vision_config.get('hidden_size', 1024)

        if actual_hidden_size != config_hidden_size:
            model.config.vision_config.update({
                'hidden_size': actual_hidden_size,
                'image_size': 320,
                'intermediate_size': actual_hidden_size * 4,
                'num_attention_heads': 24 if actual_hidden_size == 1536 else 16,
            })
            return True
    return False


def load_and_verify_model_components(model):
    """
    Verify that all model components are loaded properly:
    - Pretrained ConvNeXT CLIP
    - Trained MM projector
    - Trained binary classifier
    - Trained LLaMA
    """
    print("="*60)
    print("VERIFYING MODEL COMPONENTS")
    print("="*60)

    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        if vision_tower.use_openclip:
            print(f"✅ ConvNeXT CLIP loaded: {vision_tower.vision_tower_name}")
            print(f"   - Model type: OpenCLIP ConvNeXT")
            print(f"   - Hidden size: {vision_tower.hidden_size}")
            print(f"   - Is loaded: {vision_tower.is_loaded}")
            has_openclip_vision = hasattr(vision_tower, 'openclip_preprocess') and vision_tower.openclip_preprocess is not None
            has_openclip_processor = hasattr(vision_tower.image_processor, 'openclip_preprocess') and vision_tower.image_processor.openclip_preprocess is not None
            print(f"   - Has OpenCLIP preprocessing (vision_tower): {has_openclip_vision}")
            print(f"   - Has OpenCLIP preprocessing (image_processor): {has_openclip_processor}")
            print(f"   - OpenCLIP preprocessing available: {has_openclip_vision or has_openclip_processor}")
        else:
            print(f"⚠️  Using standard CLIP (not ConvNeXT): {vision_tower.vision_tower_name}")
    else:
        print("❌ No vision tower found!")
        return False

    if hasattr(model.get_model(), 'mm_projector') and model.get_model().mm_projector is not None:
        mm_projector = model.get_model().mm_projector
        print(f"✅ MM Projector loaded")
        print(f"   - Type: {type(mm_projector).__name__}")
        if hasattr(mm_projector, '__len__'):
            print(f"   - Layers: {len(mm_projector)}")

        first_layer_weight = None
        if hasattr(mm_projector, '0') and hasattr(mm_projector[0], 'weight'):
            first_layer_weight = mm_projector[0].weight
        elif hasattr(mm_projector, 'projector') and hasattr(mm_projector.projector[0], 'weight'):
            first_layer_weight = mm_projector.projector[0].weight

        if first_layer_weight is not None:
            weight_std = first_layer_weight.std().item()
            print(f"   - First layer weight std: {weight_std:.6f}")
            if weight_std > 0.01:
                print(f"   - ✅ Weights appear trained (std > 0.01)")
            else:
                print(f"   - ⚠️  Weights might be untrained (std <= 0.01)")
    else:
        print("❌ No MM projector found!")
        return False

    if hasattr(model.get_model(), 'binary_classifier') and model.get_model().binary_classifier is not None:
        binary_classifier = model.get_model().binary_classifier
        print(f"✅ Binary Classifier loaded")
        print(f"   - Type: {type(binary_classifier).__name__}")

        if hasattr(binary_classifier, 'fc1') and hasattr(binary_classifier.fc1, 'weight'):
            weight_std = binary_classifier.fc1.weight.std().item()
            print(f"   - FC1 layer weight std: {weight_std:.6f}")
            if weight_std > 0.01:
                print(f"   - ✅ Weights appear trained")
            else:
                print(f"   - ⚠️  Weights might be untrained")
    else:
        print("⚠️  No binary classifier found (this is OK if not doing binary classification)")

    base_model = model.get_model()
    if hasattr(base_model, 'model') and base_model.model is not None:
        print(f"✅ LLaMA model loaded")
        print(f"   - Type: {type(base_model.model).__name__}")
        print(f"   - Config: {base_model.model.config.name_or_path if hasattr(base_model.model.config, 'name_or_path') else 'Unknown'}")
    else:
        print("⚠️  LLaMA model not found at expected location, checking alternatives...")
        print(f"   - Base model type: {type(base_model).__name__}")
        print(f"   - Base model attributes: {[attr for attr in dir(base_model) if not attr.startswith('_')][:10]}")

        if hasattr(model, 'generate'):
            print("✅ Model has generate method - continuing evaluation")
        else:
            print("❌ Model cannot generate text!")
            return False

    print("="*60)
    print("MODEL COMPONENT VERIFICATION COMPLETE")
    print("="*60)
    return True


def eval_model_worker(rank, world_size, args):
    """Worker function for each GPU"""
    try:
        if world_size > 1:
            setup_distributed(rank, world_size, args.master_port)

        print(f"[GPU {rank}] Initializing ConvNeXT evaluation worker")
        print(f"[GPU {rank}] Model path: {args.model_path}")

        disable_torch_init()
        device = f'cuda:{rank}'
        torch.cuda.set_device(rank)

        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)

        print(f"[GPU {rank}] Loading pretrained model...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, args.model_base, model_name, device_map=device
        )
        print(f"[GPU {rank}] Model loaded successfully")

        config_fixed = fix_convnext_config(model)
        if config_fixed:
            print(f"[GPU {rank}] Configuration fixed for ConvNeXT compatibility")

        model = model.to(device)
        print(f"[GPU {rank}] Model moved to {device}")

        if rank == 0:
            verification_passed = load_and_verify_model_components(model)
            if not verification_passed:
                print(f"[GPU {rank}] Model component verification failed - but continuing evaluation...")
                print(f"[GPU {rank}] Attempting to proceed with available components...")

        vision_tower = model.get_vision_tower()
        if vision_tower is None:
            print(f"[GPU {rank}] Error: No vision tower found!")
            return

        try:
            with open(os.path.expanduser(args.question_file), "r") as f:
                questions = [json.loads(q) for q in f]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[GPU {rank}] Error loading question file: {e}")
            return

        questions_chunk = get_chunk(questions, world_size, rank)
        print(f"[GPU {rank}] Processing {len(questions_chunk)} questions")

        if len(questions_chunk) == 0:
            print(f"[GPU {rank}] No questions assigned, exiting")
            return

        answers_file_base = os.path.expanduser(args.answers_file)
        if world_size == 1:
            rank_answers_file = answers_file_base
        else:
            answers_dir = os.path.dirname(answers_file_base)
            answers_name = os.path.basename(answers_file_base)
            answers_name_parts = os.path.splitext(answers_name)
            rank_answers_file = os.path.join(
                answers_dir,
                f"{answers_name_parts[0]}_rank_{rank}{answers_name_parts[1]}"
            )

        answers_dir = os.path.dirname(rank_answers_file)
        os.makedirs(answers_dir, exist_ok=True)
        print(f"[GPU {rank}] Output file: {rank_answers_file}")

        processed_count = 0
        error_count = 0
        start_time = time.time()

        print(f"[GPU {rank}] Starting to process {len(questions_chunk)} questions")
        print(f"[GPU {rank}] Question IDs range: {questions_chunk[0]['question_id']} to {questions_chunk[-1]['question_id']}")

        with open(rank_answers_file, "w") as ans_file:
            for i, line in enumerate(tqdm(questions_chunk, desc=f"GPU {rank}", position=rank)):
                if i % 50 == 0:
                    print(f"[GPU {rank}] Progress: {i}/{len(questions_chunk)} ({100*i/len(questions_chunk):.1f}%) - {processed_count} successful, {error_count} errors")
                try:
                    idx = line["question_id"]
                    image_file = line["image"]
                    qs = line["text"]
                    label = line.get("label", "unknown")
                    cur_prompt = qs

                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    input_ids = tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                    ).unsqueeze(0).to(device)

                    image_path = os.path.join(args.image_folder, image_file)
                    if not os.path.exists(image_path):
                        print(f"[GPU {rank}] Warning: Image not found: {image_path}, skipping question_id: {idx}")
                        continue

                    try:
                        image = Image.open(image_path).convert('RGB')
                        image_size = image.size

                        image_tensor = process_convnext_image(image, vision_tower, model.config)

                        if image_tensor.dim() == 3:
                            image_tensor = image_tensor.unsqueeze(0)
                        image_tensor = image_tensor.to(device=device, dtype=torch.float16)

                    except Exception as e:
                        print(f"[GPU {rank}] Error processing image {image_path}: {e}, skipping question_id: {idx}")
                        error_count += 1
                        continue

                    with torch.inference_mode():
                        try:
                            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

                            try:
                                output_ids = model.generate(
                                    input_ids,
                                    images=image_tensor,
                                    image_sizes=[image_size],
                                    do_sample=True if args.temperature > 0 else False,
                                    temperature=args.temperature,
                                    max_new_tokens=args.max_new_tokens,
                                    pad_token_id=pad_token_id
                                )
                            except Exception as gen_error:
                                print(f"[GPU {rank}] Generation failed, trying fallback without image_sizes: {gen_error}")

                            decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                            if len(decoded_outputs) > 0:
                                outputs = decoded_outputs[0].strip()
                            else:
                                print(f"[GPU {rank}] Warning: No decoded output for question {idx}")
                                continue

                            prediction = "None"
                            if args.parse_binary_output:
                                if '### Answer:' in outputs:
                                    prediction = outputs.split('### Answer:')[-1].strip().lower()
                                    if 'real' in prediction:
                                        prediction = 'real'
                                    elif 'fake' in prediction:
                                        prediction = 'fake'

                                    if prediction not in ['real', 'fake']:
                                        prediction = "None"

                            ans_id = shortuuid.uuid()
                            result = {
                                "question_id": idx,
                                "prompt": cur_prompt,
                                "text": outputs,
                                "answer_id": ans_id,
                                "model_id": model_name,
                                "predict": prediction,
                                "label": label,
                                "gpu_rank": rank,
                                "preprocessing": "convnext_openclip" if hasattr(vision_tower, 'openclip_preprocess') else "fallback"
                            }

                            ans_file.write(json.dumps(result) + "\n")
                            ans_file.flush()
                            processed_count += 1

                            if processed_count % 10 == 0:
                                torch.cuda.empty_cache()

                        except Exception as e:
                            print(f"[GPU {rank}] Error during generation for question {idx}: {e}")
                            error_count += 1
                            continue

                except Exception as e:
                    print(f"[GPU {rank}] Error processing question {idx}: {e}")
                    error_count += 1
                    continue

        elapsed_time = time.time() - start_time
        print(f"[GPU {rank}] FINAL SUMMARY:")
        print(f"[GPU {rank}] - Total assigned: {len(questions_chunk)}")
        print(f"[GPU {rank}] - Successfully processed: {processed_count}")
        print(f"[GPU {rank}] - Errors encountered: {error_count}")
        print(f"[GPU {rank}] - Success rate: {100*processed_count/len(questions_chunk):.1f}%")
        print(f"[GPU {rank}] - Total time: {elapsed_time:.2f} seconds")
        print(f"[GPU {rank}] - Average time per question: {elapsed_time/max(processed_count, 1):.2f} seconds")

    except Exception as e:
        print(f"[GPU {rank}] Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if world_size > 1:
            cleanup_distributed()


def merge_results(args, world_size):
    """Merge results from all GPUs into a single file"""
    print("Merging results from all GPUs...")

    answers_file_base = os.path.expanduser(args.answers_file)
    answers_dir = os.path.dirname(answers_file_base)
    answers_name = os.path.basename(answers_file_base)
    answers_name_parts = os.path.splitext(answers_name)

    all_results = []
    total_by_preprocessing = {"convnext_openclip": 0, "fallback": 0}

    for rank in range(world_size):
        rank_file = os.path.join(
            answers_dir,
            f"{answers_name_parts[0]}_rank_{rank}{answers_name_parts[1]}"
        )

        if os.path.exists(rank_file):
            rank_results = 0
            with open(rank_file, 'r') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        all_results.append(result)
                        rank_results += 1

                        preprocessing = result.get('preprocessing', 'unknown')
                        if preprocessing in total_by_preprocessing:
                            total_by_preprocessing[preprocessing] += 1

            print(f"Loaded {rank_results} results from GPU {rank}")
        else:
            print(f"Warning: Results file for GPU {rank} not found: {rank_file}")

    all_results.sort(key=lambda x: x['question_id'])

    with open(answers_file_base, 'w') as f:
        for result in all_results:
            result.pop('gpu_rank', None)
            f.write(json.dumps(result) + "\n")

    print(f"Merged {len(all_results)} results into {answers_file_base}")
    print(f"Preprocessing methods used:")
    for method, count in total_by_preprocessing.items():
        if count > 0:
            print(f"  - {method}: {count} images")

    if args.cleanup_temp_files:
        for rank in range(world_size):
            rank_file = os.path.join(
                answers_dir,
                f"{answers_name_parts[0]}_rank_{rank}{answers_name_parts[1]}"
            )
            if os.path.exists(rank_file):
                os.remove(rank_file)
                print(f"Cleaned up temporary file: {rank_file}")


def main():
    parser = argparse.ArgumentParser(description="ConvNeXT Multi-GPU LLaVA VQA Evaluation")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the trained LLaVA model with ConvNeXT")
    parser.add_argument("--model-base", type=str, default=None,
                       help="Base model path (for LoRA models)")
    parser.add_argument("--image-folder", type=str, required=True,
                       help="Path to the image folder")
    parser.add_argument("--question-file", type=str, required=True,
                       help="Path to the question JSONL file")
    parser.add_argument("--answers-file", type=str, required=True,
                       help="Path to save evaluation results")
    parser.add_argument("--conv-mode", type=str, default="llava_v1",
                       help="Conversation mode for the model")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=None,
                       help="Top-p sampling parameter")
    parser.add_argument("--num_beams", type=int, default=1,
                       help="Number of beams for beam search")
    parser.add_argument("--max_new_tokens", type=int, default=3072,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--world-size", type=int, default=None,
                       help="Number of GPUs to use (default: all available)")
    parser.add_argument("--cleanup-temp-files", action="store_true",
                       help="Remove temporary per-GPU result files after merging")
    parser.add_argument("--parse-binary-output", action="store_true",
                       help="Parse binary classification predictions from output")
    parser.add_argument("--force-single-gpu", action="store_true",
                       help="Force single GPU mode (no distributed processing)")
    parser.add_argument("--master-port", type=int, default=None,
                       help="Master port for distributed processing (default: find free port)")

    args = parser.parse_args()

    if args.force_single_gpu:
        args.world_size = 1
        print("🔧 Forced single GPU mode")
    elif args.world_size is None:
        args.world_size = torch.cuda.device_count()

    print(f"🚀 Starting ConvNeXT LLaVA Evaluation")
    print(f"📁 Model path: {args.model_path}")
    print(f"🖼️  Image folder: {args.image_folder}")
    print(f"❓ Question file: {args.question_file}")
    print(f"📝 Output file: {args.answers_file}")
    print(f"🔧 Using {args.world_size} GPUs for inference")

    if args.world_size <= 1:
        print("ℹ️  Using single-GPU mode (no distributed processing)")

    start_time = time.time()

    try:
        if args.world_size == 1:
            print("🚀 Running in single GPU mode...")
            eval_model_worker(0, 1, args)

        else:
            print("🚀 Running in multi-GPU mode...")

            if args.master_port is None:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', 0))
                    args.master_port = s.getsockname()[1]
                print(f"🔧 Using master port: {args.master_port}")

            mp.spawn(
                eval_model_worker,
                args=(args.world_size, args),
                nprocs=args.world_size,
                join=True
            )

            merge_results(args, args.world_size)

        total_time = time.time() - start_time
        print(f"✅ Total inference time: {total_time:.2f} seconds")
        print(f"📊 Results saved to: {args.answers_file}")

    except Exception as e:
        print(f"❌ Error during multi-GPU inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
