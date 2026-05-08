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

from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer as get_megatron_optimizer_native
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
    no_weight_decay_cond=None,
    scale_lr_cond=None,
    lr_mult=1.0,
):
    # Base optimizer.
    return get_megatron_optimizer_native(
        config=config,
        model_chunks=model,
        no_weight_decay_cond=no_weight_decay_cond,
        scale_lr_cond=scale_lr_cond,
        lr_mult=lr_mult,
    )


def get_megatron_optimizer_param_scheduler(
    optimizer,
    config,
):
    """
    Get the optimizer parameter scheduler for Megatron.
    """
    if config.get("lr_decay_steps", None) is None:
        config.lr_decay_steps = config.total_training_steps
    wsd_decay_steps = None
    if config.get("lr_wsd_decay_steps", None) is not None:
        wsd_decay_steps = config.lr_wsd_decay_steps
    if config.get("lr_warmup_steps_ratio", None) is not None and (
        config.get("lr_warmup_steps", None) is None or config.lr_warmup_steps <= 0
    ):
        config.lr_warmup_steps = int(config.lr_warmup_steps_ratio * config.lr_decay_steps)

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=config.lr_warmup_init,
        max_lr=config.lr,
        min_lr=config.min_lr,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_decay_steps=config.lr_decay_steps,
        lr_decay_style=config.lr_decay_style,
        start_wd=config.weight_decay,
        end_wd=config.weight_decay,
        wd_incr_steps=config.total_training_steps,
        wd_incr_style=config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=(not config.use_checkpoint_opt_param_scheduler),
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=config.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def get_megatron_last_lr(optimizer):
    """
    Get the last learning rate from the optimizer parameter scheduler.
    """
    return optimizer.param_groups[0]["lr"]
