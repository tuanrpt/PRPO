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

import numbers

import torch
from apex.normalization.fused_layer_norm import fused_rms_norm_affine
from megatron.core import ModelParallelConfig
from torch import nn
from transformers import Qwen2Config

from verl.utils.megatron import sequence_parallel as sp_utils


class ParallelQwen2RMSNorm(nn.Module):
    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        if isinstance(config.hidden_size, numbers.Integral):
            normalized_shape = (config.hidden_size,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.variance_epsilon = config.rms_norm_eps

        if megatron_config.sequence_parallel:
            sp_utils.mark_parameter_as_sequence_parallel(self.weight)

    def forward(self, hidden_states):
        return fused_rms_norm_affine(
            input=hidden_states,
            weight=self.weight,
            normalized_shape=self.normalized_shape,
            eps=self.variance_epsilon,
            memory_efficient=True,
        )
