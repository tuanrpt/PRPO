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

import logging
import time

import ray
from cupy.cuda.nccl import NcclCommunicator, get_unique_id
from ray.util import list_named_actors


@ray.remote
class NCCLIDStore:
    def __init__(self, nccl_id):
        self._nccl_id = nccl_id

    def get(self):
        return self._nccl_id


def get_nccl_id_store_by_name(name):
    all_actors = list_named_actors(all_namespaces=True)
    matched_actors = [actor for actor in all_actors if actor.get("name", None) == name]
    if len(matched_actors) == 1:
        actor = matched_actors[0]
        return ray.get_actor(**actor)
    elif len(matched_actors) > 1:
        logging.warning("multiple actors with same name found: %s", matched_actors)
    elif len(matched_actors) == 0:
        logging.info("failed to get any actor named %s", name)
    return None


def create_nccl_communicator_in_ray(
    rank: int, world_size: int, group_name: str, max_retries: int = 100, interval_s: int = 5
):
    if rank == 0:
        nccl_id = get_unique_id()
        nccl_id_store = NCCLIDStore.options(name=group_name).remote(nccl_id)

        assert ray.get(nccl_id_store.get.remote()) == nccl_id
        communicator = NcclCommunicator(
            ndev=world_size,
            commId=nccl_id,
            rank=0,
        )
        return communicator
    else:
        for i in range(max_retries):
            nccl_id_store = get_nccl_id_store_by_name(group_name)
            if nccl_id_store is not None:
                logging.info("nccl_id_store %s got", group_name)
                nccl_id = ray.get(nccl_id_store.get.remote())
                logging.info("nccl id for %s got: %s", group_name, nccl_id)
                communicator = NcclCommunicator(
                    ndev=world_size,
                    commId=nccl_id,
                    rank=rank,
                )
                return communicator
            logging.info("failed to get nccl_id for %d time, sleep for %d seconds", i + 1, interval_s)
            time.sleep(interval_s)
