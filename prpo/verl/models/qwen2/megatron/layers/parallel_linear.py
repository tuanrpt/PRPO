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

# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py


from megatron.core import tensor_parallel


class QKVParallelLinear(tensor_parallel.ColumnParallelLinear):
    def __init__(
        self,
        input_size,
        num_heads,
        num_key_value_heads,
        head_dim,
        *,
        bias=True,
        gather_output=True,
        skip_bias_add=False,
        **kwargs,
    ):
        # Keep input parameters, and already restrict the head numbers
        self.input_size = input_size
        self.q_output_size = num_heads * head_dim
        self.kv_output_size = num_key_value_heads * head_dim
        self.head_dim = head_dim
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add

        input_size = self.input_size
        output_size = (num_heads + 2 * num_key_value_heads) * self.head_dim

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            **kwargs,
        )


class MergedColumnParallelLinear(tensor_parallel.ColumnParallelLinear):
    def __init__(
        self,
        input_size,
        gate_ouput_size,
        up_output_size,
        *,
        bias=True,
        gather_output=True,
        skip_bias_add=False,
        **kwargs,
    ):
        # Keep input parameters, and already restrict the head numbers
        self.input_size = input_size
        self.output_size = gate_ouput_size + up_output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add

        super().__init__(
            input_size=self.input_size,
            output_size=self.output_size,
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            **kwargs,
        )
