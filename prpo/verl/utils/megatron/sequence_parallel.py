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

import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu


def mark_parameter_as_sequence_parallel(parameter):
    parameter.sequence_parallel = True


def is_sequence_parallel_param(param):
    return hasattr(param, "sequence_parallel") and param.sequence_parallel


def pad_to_sequence_parallel(unpad_tokens: torch.Tensor):
    """pad the tokens such that the total length is a multiple of sp world size

    Args:
        unpad_tokens: (total_nnz, ...). Tokens after removing padding

    Returns:
        the padded tokens: (total_nnz + pad_size,...)

    """
    total_nnz = unpad_tokens.shape[0]
    sp_world_size = mpu.get_tensor_model_parallel_world_size()

    pad_size = 0 if total_nnz % sp_world_size == 0 else sp_world_size - total_nnz % sp_world_size

    if pad_size > 0:
        if unpad_tokens.ndim == 1:
            unpad_tokens = F.pad(unpad_tokens, (0, pad_size))
        elif unpad_tokens.ndim == 2:
            unpad_tokens = F.pad(unpad_tokens, (0, 0, 0, pad_size))
        else:
            raise NotImplementedError(f"Padding dim {unpad_tokens.ndim()} is not supported")

    return unpad_tokens
