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

import functools
from contextlib import contextmanager
from typing import Callable, Dict, Optional

import nvtx
import torch

from .profile import DistProfiler, ProfilerConfig


def mark_start_range(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    """Start a mark range in the profiler.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
        color (str, optional):
            The color of the range. Defaults to None.
        domain (str, optional):
            The domain of the range. Defaults to None.
        category (str, optional):
            The category of the range. Defaults to None.
    """
    return nvtx.start_range(message=message, color=color, domain=domain, category=category)


def mark_end_range(range_id: str) -> None:
    """End a mark range in the profiler.

    Args:
        range_id (str):
            The id of the mark range to end.
    """
    return nvtx.end_range(range_id)


def mark_annotate(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> Callable:
    """Decorate a function to annotate a mark range along with the function life cycle.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
        color (str, optional):
            The color of the range. Defaults to None.
        domain (str, optional):
            The domain of the range. Defaults to None.
        category (str, optional):
            The category of the range. Defaults to None.
    """

    def decorator(func):
        profile_message = message or func.__name__
        return nvtx.annotate(profile_message, color=color, domain=domain, category=category)(func)

    return decorator


@contextmanager
def marked_timer(
    name: str,
    timing_raw: Dict[str, float],
    color: str = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    """Context manager for timing with NVTX markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds NVTX markers for profiling.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
        color (Optional[str]): Color for the NVTX marker. Defaults to None.
        domain (Optional[str]): Domain for the NVTX marker. Defaults to None.
        category (Optional[str]): Category for the NVTX marker. Defaults to None.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    mark_range = mark_start_range(message=name, color=color, domain=domain, category=category)
    from .performance import _timer

    yield from _timer(name, timing_raw)
    mark_end_range(mark_range)


class NsightSystemsProfiler(DistProfiler):
    """
    Nsight system profiler. Installed in a worker to control the Nsight system profiler.
    """

    def __init__(self, rank: int, config: ProfilerConfig):
        config = config
        self.this_step: bool = False
        self.discrete: bool = config.discrete
        self.this_rank: bool = False
        if config.all_ranks:
            self.this_rank = True
        elif config.ranks is not None:
            self.this_rank = rank in config.ranks

    def start(self):
        if self.this_rank:
            self.this_step = True
            if not self.discrete:
                torch.cuda.profiler.start()

    def stop(self):
        if self.this_rank:
            self.this_step = False
            if not self.discrete:
                torch.cuda.profiler.stop()

    @staticmethod
    def annotate(
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Callable:
        """Decorate a Worker member function to profile the current rank in the current training step.

        Requires the target function to be a member function of a Worker, which has a member field `profiler` with
        NightSystemsProfiler type.

        Args:
            message (str, optional):
                The message to be displayed in the profiler. Defaults to None.
            color (str, optional):
                The color of the range. Defaults to None.
            domain (str, optional):
                The domain of the range. Defaults to None.
            category (str, optional):
                The category of the range. Defaults to None.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                profile_name = message or func.__name__

                if self.profiler.this_step:
                    if self.profiler.discrete:
                        torch.cuda.profiler.start()
                    mark_range = mark_start_range(message=profile_name, color=color, domain=domain, category=category)

                result = func(self, *args, **kwargs)

                if self.profiler.this_step:
                    mark_end_range(mark_range)
                    if self.profiler.discrete:
                        torch.cuda.profiler.stop()

                return result

            return wrapper

        return decorator
