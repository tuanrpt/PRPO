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

try:
    from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
except:
    pass
