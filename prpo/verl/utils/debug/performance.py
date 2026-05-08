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

import datetime
import inspect
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from codetiming import Timer

from verl.utils.device import get_device_id, get_torch_device
from verl.utils.logger import DecoratorLoggerBase


def _get_current_mem_info(unit: str = "GB", precision: int = 2) -> Tuple[str]:
    """Get current memory usage."""
    assert unit in ["GB", "MB", "KB"]
    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
    mem_allocated = get_torch_device().memory_allocated()
    mem_reserved = get_torch_device().memory_reserved()
    # use get_torch_device().mem_get_info to profile device memory
    # since vllm's sleep mode works below pytorch
    # see https://github.com/vllm-project/vllm/pull/11743#issuecomment-2754338119
    mem_free, mem_total = get_torch_device().mem_get_info()
    mem_used = mem_total - mem_free
    mem_allocated = f"{mem_allocated / divisor:.{precision}f}"
    mem_reserved = f"{mem_reserved / divisor:.{precision}f}"
    mem_used = f"{mem_used / divisor:.{precision}f}"
    mem_total = f"{mem_total / divisor:.{precision}f}"
    return mem_allocated, mem_reserved, mem_used, mem_total


def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = (
            f"{head}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}"
        )

        if logger is None:
            print(message)
        else:
            logger.log(msg=message, level=level)


class GPUMemoryLogger(DecoratorLoggerBase):
    """A decorator class to log GPU memory usage.

    Example:
        >>> from verl.utils.debug.performance import GPUMemoryLogger
        >>> @GPUMemoryLogger(role="actor")
        >>> def update_actor(self, batch):
        ...     # real actor update logics
        ...     return
    """

    def __init__(self, role: str, logger: logging.Logger = None, level=logging.DEBUG, log_only_rank_0: bool = True):
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
        else:
            rank = 0
        super().__init__(role, logger, level, rank, log_only_rank_0)

    def __call__(self, decorated_function: callable):
        def f(*args, **kwargs):
            return self.log(decorated_function, *args, **kwargs)

        return f

    def log(self, func, *args, **kwargs):
        name = func.__name__
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = (
            f"Before {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}"
        )
        self.logging_function(message)

        output = func(*args, **kwargs)

        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = (
            f"After {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}"
        )

        self.logging_function(message)
        return output


def log_print(ctn: Any):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno
    file_name = frame.f_code.co_filename.split("/")[-1]
    print(f"[{current_time}-{file_name}:{line_number}:{function_name}]: {ctn}")


def _timer(name: str, timing_raw: Dict[str, float]):
    """Inner function that handles the core timing logic.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


@contextmanager
def simple_timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for basic timing without NVTX markers.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    yield from _timer(name, timing_raw)


@contextmanager
def marked_timer(
    name: str,
    timing_raw: Dict[str, float],
    color: str = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    """Context manager for timing with platform markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds platform markers for profiling.
    This function is a default implementation when hardware profiler is not available.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
        color (Optional[str]): Color for the marker. Defaults to None.
        domain (Optional[str]): Domain for the marker. Defaults to None.
        category (Optional[str]): Category for the marker. Defaults to None.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    yield from _timer(name, timing_raw)


def reduce_timing(timing_raw: Dict[str, float]) -> Dict[str, float]:
    """Reduce timing information across all processes.

    This function uses distributed communication to gather and sum the timing
    information from all processes in a distributed environment.

    Args:
        timing_raw (Dict[str, float]): Dictionary containing timing information.

    Returns:
        Dict[str, float]: Reduced timing information.
    """
    if not dist.is_initialized():
        return timing_raw

    key_list, timing_list = [], []
    for key in sorted(timing_raw.keys()):
        key_list.append(key)
        timing_list.append(timing_raw[key])
    timing_list = torch.tensor(timing_list, dtype=torch.float32, device=get_device_id())
    torch.distributed.all_reduce(timing_list, op=torch.distributed.ReduceOp.AVG)
    timing_list = [tensor.item() for tensor in timing_list.to("cpu")]
    timing_generate = {key_list[i]: timing_list[i] for i in range(len(key_list))}
    return timing_generate
