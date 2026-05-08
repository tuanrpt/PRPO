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

__all__ = ["register", "get_reward_manager_cls"]

REWARD_MANAGER_REGISTRY = {}


def register(name):
    """Decorator to register a reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.
    """

    def decorator(cls):
        if name in REWARD_MANAGER_REGISTRY and REWARD_MANAGER_REGISTRY[name] != cls:
            raise ValueError(
                f"Reward manager {name} has already been registered: {REWARD_MANAGER_REGISTRY[name]} vs {cls}"
            )
        REWARD_MANAGER_REGISTRY[name] = cls
        return cls

    return decorator


def get_reward_manager_cls(name):
    """Get the reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.

    Returns:
        `(type)`: The reward manager class.
    """
    if name not in REWARD_MANAGER_REGISTRY:
        raise ValueError(f"Unknown reward manager: {name}")
    return REWARD_MANAGER_REGISTRY[name]
