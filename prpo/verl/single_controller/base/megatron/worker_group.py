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

from typing import Dict

from verl.single_controller.base import ResourcePool, WorkerGroup

from .worker import DistGlobalInfo, DistRankInfo


class MegatronWorkerGroup(WorkerGroup):
    def __init__(self, resource_pool: ResourcePool, **kwargs):
        super().__init__(resource_pool=resource_pool, **kwargs)
        self._megatron_rank_info = None
        self._megatron_global_info: DistGlobalInfo = None

    def init_megatron(self, default_megatron_kwargs: Dict = None):
        raise NotImplementedError("MegatronWorkerGroup.init_megatron should be overwritten")

    def get_megatron_rank_info(self, rank: int) -> DistRankInfo:
        assert 0 <= rank < self.world_size, f"rank must be from [0, world_size), Got {rank}"
        return self._megatron_rank_info[rank]

    @property
    def tp_size(self):
        assert self._megatron_global_info is not None, "MegatronWorkerGroup._megatron_global_info must be initialized"
        return self._megatron_global_info.tp_size

    @property
    def dp_size(self):
        assert self._megatron_global_info is not None, "MegatronWorkerGroup._megatron_global_info must be initialized"
        return self._megatron_global_info.dp_size

    @property
    def pp_size(self):
        assert self._megatron_global_info is not None, "MegatronWorkerGroup._megatron_global_info must be initialized"
        return self._megatron_global_info.pp_size

    @property
    def cp_size(self):
        assert self._megatron_global_info is not None, "MegatronWorkerGroup._megatron_global_info must be initialized"
        return self._megatron_global_info.cp_size

    def get_megatron_global_info(self):
        return self._megatron_global_info
