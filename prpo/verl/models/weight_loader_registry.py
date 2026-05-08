# Original Copyright (c) 2023 PRIME-RL (TTRL)
# Modifications Copyright (c) 2025 Tuan Nguyen
#
# This file is modified from TTRL: https://github.com/PRIME-RL/TTRL
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def get_weight_loader(arch: str):
    from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel

    _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY = {
        "LlamaForCausalLM": load_state_dict_to_megatron_gptmodel,
        "Qwen2ForCausalLM": load_state_dict_to_megatron_gptmodel,
    }

    if arch in _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY:
        return _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {arch} loader are not supported for now. Supported architectures: "
        f"{_MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY.keys()}"
    )


def get_weight_saver(arch: str):
    from verl.models.mcore.saver import (
        merge_megatron_ckpt_gptmodel,
        merge_megatron_ckpt_gptmodel_dpskv3,
        merge_megatron_ckpt_gptmodel_mixtral,
        merge_megatron_ckpt_gptmodel_qwen2_5_vl,
        merge_megatron_ckpt_gptmodel_qwen_moe,
    )

    _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY = {
        "LlamaForCausalLM": merge_megatron_ckpt_gptmodel,
        "Qwen2ForCausalLM": merge_megatron_ckpt_gptmodel,
        "MixtralForCausalLM": merge_megatron_ckpt_gptmodel_mixtral,
        "Qwen2MoeForCausalLM": merge_megatron_ckpt_gptmodel_qwen_moe,
        "Qwen2_5_VLForConditionalGeneration": merge_megatron_ckpt_gptmodel_qwen2_5_vl,
        "DeepseekV3ForCausalLM": merge_megatron_ckpt_gptmodel_dpskv3,
        "Qwen3ForCausalLM": merge_megatron_ckpt_gptmodel,
        "Qwen3MoeForCausalLM": merge_megatron_ckpt_gptmodel_qwen_moe,
    }
    if arch in _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY:
        return _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {arch} saver are not supported for now. Supported architectures: "
        f"{_MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY.keys()}"
    )
