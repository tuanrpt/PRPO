#!/usr/bin/env python3

# Copyright (c) 2025 Tuan Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Optimized LLaMA weight extraction from DX-LLaVA model.
Processes one file at a time to reduce memory usage.
"""

import os
import json
import shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
import gc
from tqdm import tqdm
import time

def extract_llama_weights_optimized(
    llava_path: str,
    output_path: str,
    exclude_patterns: list = None
):
    """Extract LLaMA weights from LLaVA safetensors files with memory optimization."""

    if exclude_patterns is None:
        exclude_patterns = [
            "mm_projector",
            "vision_tower",
            "binary_classifier",
            "clip_"
        ]

    print(f"🔄 Extracting LLaMA weights from: {llava_path}")
    print(f"📁 Output directory: {output_path}")

    os.makedirs(output_path, exist_ok=True)

    index_path = os.path.join(llava_path, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]

    llama_weight_map = {}
    for param_name, file_name in weight_map.items():
        if any(pattern in param_name for pattern in exclude_patterns):
            print(f"⏭️  Skipping: {param_name}")
            continue
        llama_weight_map[param_name] = file_name

    file_groups = {}
    for param_name, file_name in llama_weight_map.items():
        if file_name not in file_groups:
            file_groups[file_name] = []
        file_groups[file_name].append(param_name)

    print(f"📊 Will extract from {len(file_groups)} files")
    print(f"🔢 Total LLaMA parameters to extract: {len(llama_weight_map)}")

    all_weights = {}
    start_time = time.time()

    with tqdm(total=len(file_groups), desc="🔄 Extracting weights", unit="file") as pbar:
        for i, (file_name, param_names) in enumerate(file_groups.items()):
            file_start = time.time()
            pbar.set_description(f"📂 Processing {file_name}")

            file_path = os.path.join(llava_path, file_name)
            weights = load_file(file_path)

            param_count = 0
            for param_name in param_names:
                if param_name in weights:
                    all_weights[param_name] = weights[param_name]
                    param_count += 1

            del weights
            gc.collect()

            file_time = time.time() - file_start
            elapsed_total = time.time() - start_time
            avg_time_per_file = elapsed_total / (i + 1)
            remaining_files = len(file_groups) - (i + 1)
            eta = remaining_files * avg_time_per_file

            pbar.set_postfix({
                'params': f'{param_count}/{len(param_names)}',
                'time': f'{file_time:.1f}s',
                'ETA': f'{eta:.0f}s'
            })
            pbar.update(1)

    total_time = time.time() - start_time
    print(f"🔢 Total extracted parameters: {len(all_weights)}")
    print(f"⏱️ Total extraction time: {total_time:.1f}s ({total_time/60:.1f}m)")

    print("💾 Saving extracted weights...")
    save_start = time.time()
    output_file = os.path.join(output_path, "model.safetensors")

    with tqdm(desc="💾 Saving weights", unit="MB") as save_pbar:
        save_file(all_weights, output_file)
        save_pbar.update(1)

    save_time = time.time() - save_start
    print(f"💾 Saved weights to: {output_file} (took {save_time:.1f}s)")

    total_params = sum(w.numel() for w in all_weights.values())
    print(f"🧮 Parameter count: {total_params:,}")
    print(f"💾 Model size (bfloat16): ~{total_params * 2 / 1e9:.2f} GB")

    return all_weights

def create_llama_config(llava_config_path: str, output_path: str):
    """Create LLaMA config from LLaVA config."""
    with open(llava_config_path, 'r') as f:
        llava_config = json.load(f)

    llama_config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": llava_config.get("attention_bias", False),
        "attention_dropout": llava_config.get("attention_dropout", 0.0),
        "bos_token_id": llava_config.get("bos_token_id", 1),
        "eos_token_id": llava_config.get("eos_token_id", 2),
        "head_dim": llava_config.get("head_dim", 128),
        "hidden_act": llava_config.get("hidden_act", "silu"),
        "hidden_size": llava_config.get("hidden_size", 4096),
        "initializer_range": llava_config.get("initializer_range", 0.02),
        "intermediate_size": llava_config.get("intermediate_size", 11008),
        "max_position_embeddings": llava_config.get("max_position_embeddings", 4096),
        "mlp_bias": llava_config.get("mlp_bias", False),
        "model_type": "llama",
        "num_attention_heads": llava_config.get("num_attention_heads", 32),
        "num_hidden_layers": llava_config.get("num_hidden_layers", 32),
        "num_key_value_heads": llava_config.get("num_key_value_heads", 32),
        "pad_token_id": llava_config.get("pad_token_id", 0),
        "pretraining_tp": llava_config.get("pretraining_tp", 1),
        "rms_norm_eps": llava_config.get("rms_norm_eps", 1e-05),
        "rope_scaling": llava_config.get("rope_scaling", None),
        "rope_theta": llava_config.get("rope_theta", 10000.0),
        "tie_word_embeddings": llava_config.get("tie_word_embeddings", False),
        "torch_dtype": "bfloat16",
        "transformers_version": llava_config.get("transformers_version", "4.37.2"),
        "use_cache": llava_config.get("use_cache", True),
        "vocab_size": llava_config.get("vocab_size", 32000)
    }

    config_path = os.path.join(output_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(llama_config, f, indent=2)

    print(f"📝 Saved LLaMA config to: {config_path}")
    return llama_config

def copy_tokenizer_files(llava_path: str, output_path: str):
    """Copy tokenizer files and add chat template."""
    tokenizer_files = [
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]

    for file_name in tokenizer_files:
        src_path = os.path.join(llava_path, file_name)
        dst_path = os.path.join(output_path, file_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"📋 Copied {file_name}")

    tokenizer_config_path = os.path.join(output_path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, 'r') as f:
            config = json.load(f)

        config["chat_template"] = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human questions.{% for message in messages %}{% if message['role'] == 'user' %} USER: {{ message['content'] }}{% elif message['role'] == 'assistant' %} ASSISTANT: {{ message['content'] }}</s>{% endif %}{% endfor %}{% if add_generation_prompt %} ASSISTANT:{% endif %}"

        with open(tokenizer_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Added chat template to tokenizer config")

def create_generation_config(output_path: str):
    """Create generation config."""
    generation_config = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "max_length": 4096
    }

    config_path = os.path.join(output_path, "generation_config.json")
    with open(config_path, 'w') as f:
        json.dump(generation_config, f, indent=2)

    print(f"⚙️ Created generation config: {config_path}")

