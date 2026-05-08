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

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

@dataclass
class GrpoConfig:
    clip_eps: float = 0.2
    normalize_adv: bool = False
    entropy_coef: float = 0.0
    kl_coef: float = 0.0
    eps: float = 1e-8

def _group_mean(values: torch.Tensor, group_ids: torch.Tensor, num_groups: Optional[int] = None) -> torch.Tensor:
    """
    Compute the mean of `values` per group specified by `group_ids`.
    Args:
        values: (N,) tensor
        group_ids: (N,) int tensor with group indices [0..G-1]
        num_groups: Optional number of groups G (inferred if None)
    Returns:
        mean_per_item: (N,) tensor where each position i has the mean of its group
    """
    assert values.ndim == 1 and group_ids.ndim == 1
    if num_groups is None:
        num_groups = int(group_ids.max().item()) + 1
    sums = torch.zeros(num_groups, device=values.device, dtype=values.dtype)
    counts = torch.zeros(num_groups, device=values.device, dtype=values.dtype)
    sums.scatter_add_(0, group_ids, values)
    one = torch.ones_like(values)
    counts.scatter_add_(0, group_ids, one)
    means = sums / (counts + 1e-8)
    return means[group_ids]

def _group_std(values: torch.Tensor, group_ids: torch.Tensor, num_groups: Optional[int] = None) -> torch.Tensor:
    """
    Compute the std of `values` per group. (Unbiased=False)
    """
    if num_groups is None:
        num_groups = int(group_ids.max().item()) + 1
    means = _group_mean(values, group_ids, num_groups)
    diffs = values - means
    var_sums = torch.zeros(num_groups, device=values.device, dtype=values.dtype)
    counts = torch.zeros(num_groups, device=values.device, dtype=values.dtype)
    var_sums.scatter_add_(0, group_ids, diffs * diffs)
    one = torch.ones_like(values)
    counts.scatter_add_(0, group_ids, one)
    vars_ = var_sums / (counts + 1e-8)
    stds = torch.sqrt(vars_ + 1e-8)
    return stds[group_ids]

def grpo_paragraph_objective(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    config: GrpoConfig = GrpoConfig(),
) -> Dict[str, Any]:
    """
    Compute the paragraph-level GRPO surrogate objective with PPO-style clipping.
    Each paragraph i belongs to some group g = group_ids[i]. The advantage is
      A_i = R_i - mean_group(R).
    If normalize_adv=True, then within each group: A_i <- (A_i - mean(A))/std(A).

    Args:
        new_logprobs: (N,) tensor of current policy log-prob for each paragraph
        old_logprobs: (N,) tensor of behavior policy log-prob for each paragraph
        rewards: (N,) tensor of scalar reward for each paragraph
        group_ids: (N,) int tensor of group indices [0..G-1] indicating which group each paragraph belongs to
        config: GrpoConfig with clip_eps, normalize_adv, entropy_coef, kl_coef

    Returns:
        dict with:
            'loss': scalar tensor to minimize
            'policy_loss': scalar tensor
            'clipfrac': fraction of items where clipping was active
            'advantages': (N,) tensor (detached) advantages used
            'ratio': (N,) tensor likelihood ratios (detached)
            'kl': scalar tensor (mean reverse-KL if kl_coef>0)
    """
    assert new_logprobs.shape == old_logprobs.shape == rewards.shape == group_ids.shape
    assert new_logprobs.ndim == 1, "Inputs must be (N,) vectors"

    mean_r = _group_mean(rewards, group_ids)

    advantages = rewards - mean_r
    if config.normalize_adv:
        std_r = _group_std(rewards, group_ids)
        advantages = advantages / (std_r + config.eps)

    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantages
    policy_loss = -torch.mean(torch.minimum(unclipped, clipped))

    kl = torch.mean(old_logprobs - new_logprobs)
    loss = policy_loss + config.kl_coef * kl

    out = {
        "loss": loss,
        "policy_loss": policy_loss,
        "advantages": advantages.detach(),
        "ratio": ratio.detach(),
        "kl": kl.detach(),
        "clipfrac": torch.mean((unclipped != clipped).float()).detach(),
    }

    return out

def add_entropy_bonus(loss: torch.Tensor, entropies: torch.Tensor, entropy_coef: float) -> torch.Tensor:
    """
    Adds an entropy bonus (subtracts mean entropy times coefficient) to the loss.
    Args:
        loss: scalar loss tensor
        entropies: (N,) tensor of per-paragraph entropies
        entropy_coef: float
    Returns:
        new scalar loss
    """
    if entropy_coef == 0.0:
        return loss
    return loss - entropy_coef * entropies.mean()
