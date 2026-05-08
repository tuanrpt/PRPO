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

from megatron.core import ModelParallelConfig, tensor_parallel
from megatron.core import parallel_state as mpu
from torch import nn
from transformers.activations import ACT2FN

from verl.models.llama.megatron.layers.parallel_linear import MergedColumnParallelLinear
from verl.utils.megatron import tensor_parallel as tp_utils


class ParallelLlamaMLP(nn.Module):
    def __init__(self, config, megatron_config: ModelParallelConfig = None) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # The weight is only [hidden_size, intermediate_size // model_parallel_world_size]

        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        if megatron_config is not None:
            assert column_kwargs.get("config", False), "must have ModelParallelConfig"
            assert row_kwargs.get("config", False), "must have ModelParallelConfig"
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)

        tp_size = mpu.get_tensor_model_parallel_world_size()

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            gate_ouput_size=self.intermediate_size,
            up_output_size=self.intermediate_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )
        self.gate_size = self.intermediate_size // tp_size

        self.down_proj = tensor_parallel.RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            **row_kwargs,
        )

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate_up = self.gate_up_proj(x)[0]
        gate, up = gate_up.split(self.gate_size, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)[0]
