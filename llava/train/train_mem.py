# Original Copyright (c) 2023 Haotian Liu (LLaVA)
# Modifications Copyright (c) 2025 Tuan Nguyen
#
# This file is modified from LLaVA: https://github.com/haotian-liu/LLaVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from llava.train.train import train

def get_best_attention_backend():
    if not torch.cuda.is_available():
        print("[WARN] No GPU detected. Falling back to eager attention.")
        return "eager"

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)

    print(f"[INFO] Detected GPU: {device_name} (Compute capability: {capability})")

    if capability[0] >= 8:
        print("[INFO] Using flash_attention_2 for best performance.")
        return "flash_attention_2"

    print("[INFO] flash_attention_2 not supported on this GPU. Falling back to sdpa.")
    return "sdpa"

if __name__ == "__main__":
    attn_backend = get_best_attention_backend()
    train(attn_implementation=attn_backend)
