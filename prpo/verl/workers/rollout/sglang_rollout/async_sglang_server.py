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

import asyncio
import logging
from typing import Any, Dict, List

import ray
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse

from verl.workers.rollout.async_server import AsyncServerBase

logger = logging.getLogger(__file__)


@ray.remote(num_cpus=1)
class AsyncSglangServer(AsyncServerBase):
    def __init__(self, config: DictConfig, dp_size: int, dp_rank: int, wg_prefix: str):
        super().__init__()
        self.config = config.actor_rollout_ref
        self._tp_size = self.config.rollout.get("tensor_model_parallel_size", 1)
        self._dp_size = dp_size
        self._dp_rank = dp_rank
        self.wg_prefix = wg_prefix
        self.workers = []
        self.master_worker = None

    async def init_engine(self):
        if self.workers:
            # avoid init twice
            return
        all_actors = ray.util.list_named_actors(all_namespaces=True)
        matched_actors = [
            actor for actor in all_actors if actor.get("name", None).startswith(self.wg_prefix + "WorkerDict_")
        ]

        for matched_actor in matched_actors:
            fields = matched_actor["name"].split(":")
            assert len(fields) == 2, f"invalid actor name: {matched_actor['name']}"
            pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])

            if (self._dp_size * pg_index + local_rank) // self._tp_size == self._dp_rank:
                worker = ray.get_actor(**matched_actor)
                self.workers.append(worker)
                if (self._dp_size * pg_index + local_rank) / self._tp_size == self._dp_rank:
                    self.master_worker = worker

    async def chat_completion(self, raw_request: Request):
        request = await raw_request.json()

        # only send request to master worker in tp rank 0
        output_future = self.master_worker.chat_completion.remote(request)
        [outputs] = await asyncio.gather(output_future)
        return JSONResponse(outputs)

    async def generate(self, prompt_ids: List[int], sampling_params: Dict[str, Any], request_id: str) -> List[int]:
        return await self.master_worker.generate.remote(prompt_ids, sampling_params, request_id)

    async def wake_up(self):
        if not self.config.rollout.free_cache_engine:
            return

        tasks = [worker.wake_up.remote() for worker in self.workers]
        if tasks:
            await asyncio.gather(*tasks)

    async def sleep(self):
        if not self.config.rollout.free_cache_engine:
            return

        tasks = [worker.sleep.remote() for worker in self.workers]
        if tasks:
            await asyncio.gather(*tasks)
