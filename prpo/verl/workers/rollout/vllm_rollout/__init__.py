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

import os
from importlib.metadata import PackageNotFoundError, version

from .vllm_rollout_spmd import vLLMAsyncRollout, vLLMRollout  # noqa: F401
from .functional_vllm_embeddings_rollout import FunctionalvLLMEmbeddingsRollout  # noqa: F401


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


vllm_package_name = "vllm"
vllm_package_version = get_version(vllm_package_name)
if vllm_package_version is None:
    raise PackageNotFoundError(
        "To use vllm rollout, please ensure the 'vllm' package is properly installed. See "
        "https://verl.readthedocs.io/en/latest/start/install.html for more details"
    )

if "ROCM_PATH" in os.environ:
    import re

    match = re.match(r"(\d+\.\d+\.?\d*)", vllm_package_version)
    if match:
        vllm_package_version = match.group(1)
    else:
        raise ValueError(f"Warning: Could not parse version format: {vllm_package_version}")

vllm_mode = "spmd"
