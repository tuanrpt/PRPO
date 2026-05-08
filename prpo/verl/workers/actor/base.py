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
The base class for Actor
"""

from abc import ABC, abstractmethod
from typing import Dict

import torch

from verl import DataProto

__all__ = ["BasePPOActor"]


class BasePPOActor(ABC):
    def __init__(self, config):
        """The base class for PPO actor

        Args:
            config (DictConfig): a config passed to the PPOActor. We expect the type to be
                DictConfig (https://omegaconf.readthedocs.io/), but it can be any namedtuple in general.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute logits given a batch of data.

        Args:
            data (DataProto): a batch of data represented by DataProto. It must contain key ```input_ids```,
                ```attention_mask``` and ```position_ids```.

        Returns:
            DataProto: a DataProto containing the key ```log_probs```


        """
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> Dict:
        """Update the policy with an iterator of DataProto

        Args:
            data (DataProto): an iterator over the DataProto that returns by
                ```make_minibatch_iterator```

        Returns:
            Dict: a dictionary contains anything. Typically, it contains the statistics during updating the model
            such as ```loss```, ```grad_norm```, etc,.

        """
        pass
