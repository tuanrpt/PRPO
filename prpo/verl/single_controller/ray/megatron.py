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

from typing import Dict, Optional

import ray

from verl.single_controller.base.megatron.worker import DistGlobalInfo, DistRankInfo
from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

from .base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


# NOTE(sgm): for open-source megatron-core
class NVMegatronRayWorkerGroup(RayWorkerGroup, MegatronWorkerGroup):
    """
    MegatronWorkerGroup will query each worker of its megatron rank info and store it inside the WorkerGroup
    so that the dispatcher can use it to dispatch data.
    """

    def __init__(self, resource_pool: RayResourcePool, ray_cls_with_init: RayClassWithInitArgs, **kwargs):
        """
        Initialize the NVMegatronRayWorkerGroup.

        Args:
            resource_pool (RayResourcePool): The resource pool containing worker resources
            ray_cls_with_init (RayClassWithInitArgs): The Ray class with initialization arguments
            **kwargs: Additional keyword arguments to pass to the parent class
        """
        super().__init__(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, **kwargs)
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(method_name="get_megatron_rank_info")
        self._megatron_global_info: DistGlobalInfo = ray.get(
            self.execute_rank_zero_async(method_name="get_megatron_global_info")
        )


class MegatronRayWorkerGroup(RayWorkerGroup, MegatronWorkerGroup):
    """
    MegatronWorkerGroup will query each worker of its megatron rank info and store it inside the WorkerGroup
    so that the dispatcher can use it to dispatch data.
    """

    def __init__(
        self,
        resource_pool: RayResourcePool,
        ray_cls_with_init: RayClassWithInitArgs,
        default_megatron_kwargs: Dict = None,
        **kwargs,
    ):
        super().__init__(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            default_megatron_kwargs=default_megatron_kwargs,
            **kwargs,
        )
        self.init_megatron(default_megatron_kwargs=default_megatron_kwargs)
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(method_name="get_megatron_rank_info")
        self._megatron_global_info: DistGlobalInfo = ray.get(
            self.execute_rank_zero_async(method_name="get_megatron_global_info")
        )

    def init_megatron(self, default_megatron_kwargs: Optional[Dict] = None):
        # after super, we will call init of each worker
        if not self._is_init_with_detached_workers:
            # only init_megatron if the WorkerGroup is created from scratch
            self.execute_all_sync(method_name="init_megatron", default_megatron_kwargs=default_megatron_kwargs)
