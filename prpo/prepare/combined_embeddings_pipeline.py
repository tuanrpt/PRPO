#!/usr/bin/env python3

# Copyright (c) 2025 Tuan Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Combined pipeline for embeddings processing:
1. Precompute embeddings from model and data
2. Create parquet dataset from embeddings

Usage:
    python combined_embeddings_pipeline.py --step precompute --model-path <path> --data-path <path> --image-folder <path> --output-dir <path>
    python combined_embeddings_pipeline.py --step parquet --embeddings-dir <path> --domain <name> --output-dir <path>
    python combined_embeddings_pipeline.py --step all --model-path <path> --data-path <path> --image-folder <path> --embeddings-dir <path> --domain <name> --output-dir <path>
"""

import torch
import json
import os
import pickle
import glob
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# Import LLaVA components
from llava.mm_utils import get_model_name_from_path
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def precompute_inputs_embeds_for_dataset(
    model_path,
    data_path,
    image_folder,
    output_dir,
    domain_name=None,
    batch_size=8,
    conv_mode="llava_v1"
):
    """Step 1: Precompute embeddings from model and data"""
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava import conversation as conversation_lib

    print("=" * 50)
    print("STEP 1: PRECOMPUTING EMBEDDINGS")
    print("=" * 50)

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )

    if hasattr(model.model, 'image_newline'):
        print(f"✓ image_newline parameter found: shape={model.model.image_newline.shape}")
        print(f"  Parameter requires_grad: {model.model.image_newline.requires_grad}")
    else:
        print("⚠ image_newline parameter NOT found in loaded model")

    model.eval()
    conversation_lib.default_conversation = conversation_lib.conv_templates[conv_mode]

    if domain_name is None:
        domain_name = os.path.basename(os.path.dirname(data_path))

    print(f"🏷️ Using domain name: {domain_name}")

    try:
        with open(os.path.expanduser(data_path), "r") as f:
            training_data = [json.loads(q) for q in f]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading question file: {e}")
        return None

    # Create embeddings subdirectory structure
    domain_output_dir = os.path.join(output_dir, domain_name, "embeddings")
    os.makedirs(domain_output_dir, exist_ok=True)

    print(f"📁 Processing {len(training_data)} samples")
    print(f"💾 Output directory: {domain_output_dir}")

    for batch_start in tqdm(range(0, len(training_data), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(training_data))
        batch_data = training_data[batch_start:batch_end]

        batch_inputs_embeds = []
        batch_attention_masks = []
        batch_labels = []
        batch_metadata = []

        for sample in batch_data:
            try:
                idx = sample["question_id"]
                image_file = sample["image"]
                qs = sample["text"]
                label = sample["label"]

                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs

                conv = conversation_lib.default_conversation.copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                enhanced_human_prompt = prompt.replace(
                    "Provide your reasoning first, then conclude with a letter answer.",
                    "Provide detailed, step-by-step reasoning focusing on specific visual artifacts, then conclude with 'Answer: real' or 'Answer: fake'."
                )

                input_ids = tokenizer_image_token(
                    enhanced_human_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                ).unsqueeze(0).cuda()

                image_path = image_file
                if not os.path.exists(image_path):
                    print(f"⚠ Warning: Image not found: {image_path}, skipping question_id: {idx}")
                    continue

                image = Image.open(image_path).convert('RGB')

                try:
                    if hasattr(image_processor, 'image_mean'):
                        image_tensor = process_images([image], image_processor, model.config)[0]
                    else:
                        if hasattr(image_processor, '__call__'):
                            image_tensor = image_processor(image)
                            if not isinstance(image_tensor, torch.Tensor):
                                image_tensor = torch.tensor(image_tensor)
                        else:
                            image_tensor = process_images([image], image_processor, model.config)[0]
                except Exception as e:
                    print(f"❌ Error processing image {image_path}: {e}")
                    continue

                images = image_tensor.unsqueeze(0).half().cuda()
                image_sizes = [image.size]

                with torch.inference_mode():
                    _, prepared_position_ids, prepared_attention_mask, _, prepared_inputs_embeds, prepared_labels = model.prepare_inputs_labels_for_multimodal(
                        input_ids,
                        None,
                        torch.ones_like(input_ids),
                        None,
                        input_ids.clone(),
                        images,
                        image_sizes
                    )

                image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
                if image_token_mask.any():
                    image_token_positions = image_token_mask.nonzero(as_tuple=True)[1].cpu().tolist()
                else:
                    image_token_positions = []

                batch_inputs_embeds.append(prepared_inputs_embeds.cpu())
                batch_attention_masks.append(prepared_attention_mask.cpu())
                batch_labels.append(prepared_labels.cpu())

                batch_metadata.append({
                    'question_id': idx,
                    'text': qs,
                    'image_path': image_file,
                    'label': label,
                    'original_input_length': input_ids.shape[1],
                    'final_embed_length': prepared_inputs_embeds.shape[1],
                    'image_token_positions': image_token_positions
                })

                del input_ids, images, prepared_inputs_embeds, prepared_labels
                if prepared_attention_mask is not None:
                    del prepared_attention_mask
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Error processing sample: {e}")
                continue

        if not batch_inputs_embeds:
            print(f"⚠ Warning: Empty batch {batch_start}-{batch_end}, skipping...")
            continue

        batch_file = os.path.join(domain_output_dir, f"batch_{batch_start:06d}_{batch_end:06d}.pkl")
        batch_output = {
            'inputs_embeds': batch_inputs_embeds,
            'attention_masks': batch_attention_masks,
            'labels': batch_labels,
            'metadata': batch_metadata
        }

        with open(batch_file, 'wb') as f:
            pickle.dump(batch_output, f)

        print(f"💾 Saved batch {batch_start}-{batch_end} with {len(batch_inputs_embeds)} samples to {batch_file}")

        del batch_inputs_embeds, batch_attention_masks, batch_labels, batch_metadata

    dataset_info = {
        'total_samples': len(training_data),
        'total_batches': (len(training_data) + batch_size - 1) // batch_size,
        'batch_size': batch_size,
        'model_path': model_path,
        'data_path': data_path,
        'image_folder': image_folder
    }

    # Save dataset_info.json in the parent directory (not in embeddings subfolder)
    parent_dir = os.path.dirname(domain_output_dir)
    with open(os.path.join(parent_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"✅ Pre-computation complete! Saved to {domain_output_dir}")
    print(f"📊 Total samples processed: {len(training_data)}")
    return output_dir  # Return output_dir for parquet step


def _tensor_to_serializable(tensor):
    """Convert tensor to serializable format for parquet"""
    if torch.is_tensor(tensor):
        if tensor.dtype == torch.bfloat16:
            return tensor.detach().cpu().numpy().view(np.uint16).tobytes()
        else:
            numpy_array = tensor.detach().cpu().numpy()
            return numpy_array.tobytes()
    else:
        if isinstance(tensor, np.ndarray):
            return tensor.tobytes()
        return tensor


def _deserialize_tensor(byte_data, tensor_info):
    """Deserialize tensor from parquet format"""
    if tensor_info is None or tensor_info.get("shape") is None:
        return byte_data

    shape = tensor_info["shape"]
    dtype_str = tensor_info["dtype"]

    if hasattr(shape, 'tolist'):
        shape = shape.tolist()
    elif not isinstance(shape, (list, tuple)):
        shape = list(shape) if hasattr(shape, '__iter__') else [shape]

    if "bfloat16" in dtype_str:
        try:
            uint16_array = np.frombuffer(byte_data, dtype=np.uint16).reshape(shape)
            uint16_array = uint16_array.copy()
            tensor = torch.from_numpy(uint16_array).view(torch.bfloat16)
            return tensor
        except Exception as e:
            try:
                float_array = np.frombuffer(byte_data, dtype=np.float32).reshape(shape)
                float_array = float_array.copy()
                tensor = torch.from_numpy(float_array)
                return tensor
            except Exception as e2:
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
        raise e


class EmbeddingsRLHFDataset(Dataset):
    """Dataset for loading precomputed embeddings"""

    def __init__(
        self,
        embeddings_dir: str,
        domain: str = "ddim",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_prompt_length: Optional[int] = None,
        max_response_length: Optional[int] = 1024,
        filter_overlong_prompts: bool = True,
        suffix_prompt: str = None
    ):
        self.embeddings_dir = embeddings_dir
        self.domain = domain
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.filter_overlong_prompts = filter_overlong_prompts
        self.suffix_prompt = suffix_prompt if suffix_prompt else ""
        self.data_samples = self._load_embeddings()
        print(f"📦 Loaded {len(self.data_samples)} samples from {embeddings_dir}/{domain}")

    def _load_embeddings(self) -> List[Dict]:
        """Load embeddings from pickle files"""
        # Look for embeddings in the embeddings subdirectory
        embeddings_dir = os.path.join(self.embeddings_dir, self.domain, "embeddings")
        if not os.path.exists(embeddings_dir):
            # Fallback to old structure
            embeddings_dir = os.path.join(self.embeddings_dir, self.domain)
            if not os.path.exists(embeddings_dir):
                raise FileNotFoundError(f"Domain directory not found: {embeddings_dir}")

        batch_files = sorted(glob.glob(os.path.join(embeddings_dir, "batch_*.pkl")))
        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {embeddings_dir}")

        print(f"🔄 Loading {len(batch_files)} batch files from {embeddings_dir}")

        all_samples = []

        for batch_file in batch_files:
            try:
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)

                inputs_embeds = batch_data['inputs_embeds']
                attention_masks = batch_data['attention_masks']
                labels = batch_data['labels']
                metadata = batch_data['metadata']

                for i in range(len(inputs_embeds)):
                    sample_data = {
                        'inputs_embeds': inputs_embeds[i],
                        'attention_mask': attention_masks[i],
                        'labels': labels[i],
                        'metadata': metadata[i]
                    }

                    if self.filter_overlong_prompts and self.max_prompt_length:
                        if inputs_embeds[i].shape[0] > self.max_prompt_length:
                            continue

                    all_samples.append(sample_data)
            except Exception as e:
                print(f"⚠ Warning: Failed to load {batch_file}: {e}")
                continue

        return all_samples

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get sample in TTRL format"""
        sample = self.data_samples[idx]
        metadata = sample['metadata']

        prompt_text = "<image>\nAnalyze this image for any signs of digital manipulation or artificial generation and determine if it is real or fake. Provide detailed step-by-step reasoning analyzing specific visual artifacts, then conclude with '### Answer: real' or '### Answer: fake'."

        prompt = [
            {
                "role": "user",
                "content": prompt_text
            }
        ]

        reward_model = {
            "style": "rule",
            "ground_truth": metadata['label']
        }

        ttrl_sample = {
            "prompt": prompt,
            "reward_model": reward_model,
            "ability": "deepfake_detection",
            "data_source": f"llava-{self.domain}",
            "extra_info": {
                "split": "train",
                "index": f"llava-{self.domain}-{metadata['question_id']}",
                "image_path": metadata.get('image_path', ''),
                "original_length": metadata.get('original_input_length', 0),
                "image_token_positions": metadata.get('image_token_positions', []),
                "inputs_embeds": sample['inputs_embeds'],
                "embedding_attention_mask": sample['attention_mask'],
                "has_embeddings": True
            },
            "original_labels": sample['labels']
        }

        return ttrl_sample


def create_embeddings_parquet_data(
    embeddings_dir: str,
    domain: str,
    output_dir: str,
    suffix_prompt: str = None
):
    """Step 2: Create parquet dataset from embeddings"""
    print("=" * 50)
    print("STEP 2: CREATING PARQUET DATASET")
    print("=" * 50)

    dataset = EmbeddingsRLHFDataset(
        embeddings_dir=embeddings_dir,
        domain=domain,
        suffix_prompt=suffix_prompt
    )

    print(f"🔄 Converting {len(dataset)} samples to parquet format...")
    parquet_data = []

    for i in tqdm(range(len(dataset)), desc="Converting samples"):
        sample = dataset[i]

        def get_tensor_info(tensor):
            if torch.is_tensor(tensor):
                return {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype)
                }
            elif isinstance(tensor, np.ndarray):
                return {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype)
                }
            else:
                return {"shape": None, "dtype": "unknown"}

        extra_info_with_dtypes = sample["extra_info"].copy()
        extra_info_with_dtypes["tensor_info"] = {
            "inputs_embeds": get_tensor_info(sample["extra_info"]["inputs_embeds"]),
            "attention_mask": get_tensor_info(sample["extra_info"]["embedding_attention_mask"]),
            "original_labels": get_tensor_info(sample["original_labels"])
        }

        # Remove tensor data from extra_info to avoid duplication
        extra_info_with_dtypes.pop("inputs_embeds", None)
        extra_info_with_dtypes.pop("embedding_attention_mask", None)
        extra_info_with_dtypes.pop("has_embeddings", None)

        parquet_sample = {
            "prompt": sample["prompt"],
            "reward_model": sample["reward_model"],
            "ability": sample["ability"],
            "data_source": sample["data_source"],
            "extra_info": extra_info_with_dtypes,
            "inputs_embeds": _tensor_to_serializable(sample["extra_info"]["inputs_embeds"]),
            "attention_mask": _tensor_to_serializable(sample["extra_info"]["embedding_attention_mask"]),
            "original_labels": _tensor_to_serializable(sample["original_labels"])
        }

        parquet_data.append(parquet_sample)

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(parquet_data)

    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create train/test split
    train_df = df
    val_df = df.sample(n=min(20, len(df)), random_state=42)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(test_path, index=False)

    print(f"✅ Created parquet files:")
    print(f"  📁 Train: {train_path} ({len(train_df)} samples)")
    print(f"  📁 Val: {test_path} ({len(val_df)} samples)")

    # Create embeddings mapping
    embeddings_mapping = {}
    for i, sample in enumerate(dataset.data_samples):
        question_id = sample['metadata']['question_id']
        embeddings_mapping[question_id] = {
            'sample_idx': i,
            'inputs_embeds_shape': sample['inputs_embeds'].shape,
            'attention_mask_shape': sample['attention_mask'].shape
        }

    mapping_path = os.path.join(output_dir, "embeddings_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(embeddings_mapping, f, indent=2)

    print(f"  📋 Mapping: {mapping_path}")

    # Clean up embeddings folder to save disk space
    embeddings_folder = os.path.join(embeddings_dir, domain, "embeddings")
    if os.path.exists(embeddings_folder):
        try:
            # Calculate space to be freed
            folder_size = sum(os.path.getsize(os.path.join(embeddings_folder, f))
                            for f in os.listdir(embeddings_folder)
                            if os.path.isfile(os.path.join(embeddings_folder, f)))
            folder_size_mb = folder_size / (1024 * 1024)

            shutil.rmtree(embeddings_folder)
            print(f"🧹 Cleaned up embeddings folder (freed {folder_size_mb:.1f} MB)")
        except Exception as e:
            print(f"⚠ Warning: Failed to remove embeddings folder: {e}")

    return train_path, test_path, mapping_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Combined embeddings pipeline")
    parser.add_argument("--step", type=str, choices=["precompute", "parquet", "all"], required=True,
                       help="Which step to run: precompute, parquet, or all")

    # Precompute arguments
    parser.add_argument("--model-path", type=str, help="Path to trained LLaVA model")
    parser.add_argument("--data-path", type=str, help="Path to training data JSON file")
    parser.add_argument("--image-folder", type=str, help="Path to images folder")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation mode")

    # Parquet arguments
    parser.add_argument("--embeddings-dir", type=str, help="Directory containing embeddings")
    parser.add_argument("--domain", type=str, help="Domain name")
    parser.add_argument("--suffix-prompt", type=str, default="", help="Suffix prompt")

    # Common arguments
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    if args.step in ["precompute", "all"]:
        if not all([args.model_path, args.data_path, args.image_folder]):
            print("❌ Error: --model-path, --data-path, and --image-folder are required for precompute step")
            return

        embeddings_output = precompute_inputs_embeds_for_dataset(
            model_path=args.model_path,
            data_path=args.data_path,
            image_folder=args.image_folder,
            output_dir=args.output_dir,
            domain_name=args.domain,
            batch_size=args.batch_size,
            conv_mode=args.conv_mode
        )

        if args.step == "all" and embeddings_output:
            # Set embeddings_dir for the parquet step
            args.embeddings_dir = embeddings_output  # This is the parent directory containing domain folder
            if not args.domain:
                args.domain = os.path.basename(embeddings_output)

    if args.step in ["parquet", "all"]:
        if not all([args.embeddings_dir, args.domain]):
            print("❌ Error: --embeddings-dir and --domain are required for parquet step")
            return

        # Put parquet files in the domain directory (not nested)
        parquet_output_dir = os.path.join(args.embeddings_dir, args.domain)
        create_embeddings_parquet_data(
            embeddings_dir=args.embeddings_dir,
            domain=args.domain,
            output_dir=parquet_output_dir,
            suffix_prompt=args.suffix_prompt
        )

    print("🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main()