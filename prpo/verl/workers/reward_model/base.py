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
The base class for reward model
"""

from abc import ABC, abstractmethod

from verl import DataProto


class BasePPORewardModel(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def compute_reward(self, data: DataProto) -> DataProto:
        """Computing reward given input_ids. The transformers should output a tensor with shape
           [batch_size, sequence_length], and the value at [EOS] mask should be gathered.

        Args:
            data: must contain keys "input_ids", "attention_mask" and "position_ids".
                - input_ids: [batch_size, sequence_length]
                - attention_mask: [batch_size, sequence_length]
                - position_ids: [batch_size, sequence_length]

        Returns: a data pass protocol containing "reward". Only the [EOS] position contains the reward.
            Other position should have zero reward. Note that this may change in the future if we use
            dense reward. So, we leave the interface for general case.
            - reward: [batch_size, sequence_length].

        """
        pass
