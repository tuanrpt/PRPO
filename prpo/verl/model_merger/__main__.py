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

"""
This module is used to merge huggingface model and test verl checkpoints from FSDP and Megatron backends.

To merge FSDP checkpoints:
```sh
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```

To merge Megatron checkpoints:
```sh
python -m verl.model_merger merge \
    --backend megatron \
    --tie-word-embedding \
    --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor \
    --target_dir /path/to/merged_hf_model
```

For more details, please refer to documentation:
https://verl.readthedocs.io/en/latest/advance/checkpoint.html#convert-fsdp-and-megatron-checkpoints-to-huggingface-format-model
"""

from .base_model_merger import generate_config_from_args, parse_args


def main():
    args = parse_args()
    config = generate_config_from_args(args)
    print(f"config: {config}")

    if config.backend == "fsdp":
        from .fsdp_model_merger import FSDPModelMerger

        merger = FSDPModelMerger(config)
    elif config.backend == "megatron":
        from .megatron_model_merger import MegatronModelMerger

        merger = MegatronModelMerger(config)
    else:
        raise NotImplementedError(f"Unknown backend: {config.backend}")

    merger.merge_and_save()
    merger.cleanup()


if __name__ == "__main__":
    main()
