#!/usr/bin/env python3

# Copyright (c) 2025 Tuan Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from verl.trainer.main_ppo import main as ttrl_main

from verl.utils.reward_score.deepfake_detection_prpo_standalone import reward_func
PRPO_REWARD_AVAILABLE = True

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PRPO'))
    from grpo_paragraph import grpo_paragraph_objective, GrpoConfig
    PRPO_CORE_AVAILABLE = True
except ImportError:
    PRPO_CORE_AVAILABLE = False


class PRPOTrainer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.prpo_enabled = config.get('prpo', {}).get('enable', False) and PRPO_CORE_AVAILABLE and PRPO_REWARD_AVAILABLE

    def enhance_config_for_prpo(self, config: DictConfig) -> DictConfig:
        if not self.prpo_enabled:
            return config

        if PRPO_REWARD_AVAILABLE:
            config.custom_reward_function.path = "../verl/utils/reward_score/deepfake_detection_prpo_standalone.py"

        prpo_config = config.get('prpo', {})

        if 'train_batch_size' not in config.data:
            config.data.train_batch_size = max(1, config.data.get('train_batch_size', 8) // 2)

        if hasattr(config.actor_rollout_ref, 'actor') and hasattr(config.actor_rollout_ref.actor, 'optim'):
            original_lr = config.actor_rollout_ref.actor.optim.get('lr', 5e-7)
            config.actor_rollout_ref.actor.optim.lr = min(original_lr, 3e-7)

        return config


def setup_environment():
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    if os.getenv('ENABLE_PRPO', 'false').lower() == 'true':
        os.environ['PRPO_ENABLED'] = 'true'


@hydra.main(version_base=None, config_path="../verl/trainer/config", config_name="ppo_trainer_ttrl_prpo")
def main(config: DictConfig) -> None:
    setup_environment()

    prpo_trainer = PRPOTrainer(config)
    enhanced_config = prpo_trainer.enhance_config_for_prpo(config)

    if prpo_trainer.prpo_enabled:
        if not PRPO_CORE_AVAILABLE:
            enhanced_config.prpo.enable = False
        elif not PRPO_REWARD_AVAILABLE:
            enhanced_config.prpo.enable = False

    if prpo_trainer.prpo_enabled:
        from omegaconf import OmegaConf
        OmegaConf.set_struct(enhanced_config, False)
        enhanced_config.monitoring = enhanced_config.get('monitoring', {})
        enhanced_config.monitoring.update({
            'track_paragraph_rewards': True,
            'track_clip_similarities': True,
            'log_prpo_metrics': True
        })
        OmegaConf.set_struct(enhanced_config, True)

    result = ttrl_main(enhanced_config)
    return result


if __name__ == "__main__":
    main()