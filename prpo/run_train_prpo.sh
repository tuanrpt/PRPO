#!/bin/bash

# Copyright (c) 2025 Tuan Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=$(nvidia-smi --list-gpus | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
else
    N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export VLLM_USE_V1=1
export CUDA_LAUNCH_BLOCKING=0
export DEBUG_BATCH_RESPONSES=${DEBUG_BATCH_RESPONSES:-false}
export USE_STRUCTURED_RESPONSES=${USE_STRUCTURED_RESPONSES:-true}
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export PRPO_OPTIMIZED=true
export CLIP_MULTI_GPU=true
export YAKE_PARALLEL=true
export OMP_NUM_THREADS=$(($(nproc)/2))
export CLIP_CACHE_SIZE=1000

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_DISABLE_STRICT_CUDA_VALIDATION=1
export RAY_FORCE_GPU_VISIBILITY=1

unset VLLM_ATTENTION_BACKEND
export VLLM_DISABLE_COMPILATION=1
export TORCH_COMPILE_DISABLE=1

export ENABLE_PRPO=true
export PRPO_CLIP_EPS=0.2
export PRPO_NORMALIZE_ADV=true
export PRPO_KL_COEF=0.01
export PRPO_ENTROPY_COEF=0.0
export ALPHA_CLIP=${ALPHA_CLIP:-0.5}
export BETA_CONSISTENCY=${BETA_CONSISTENCY:-0.5}

# Set PYTHONPATH for Ray workers to find verl module
export PYTHONPATH="${PWD}:${PYTHONPATH}"

DOMAIN=${DOMAIN:-"ddim"}

LLAMA_MODEL_PATH=${LLAMA_MODEL_PATH:-"/path/to/dx_llava/model"}
TRAIN_FILES=${TRAIN_FILES:-"data/${DOMAIN}/train.parquet"}
VAL_FILES=${VAL_FILES:-"data/${DOMAIN}/test.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"../checkpoints/dx_llava_prpo_${DOMAIN}"}

EXPERIMENT_NAME="prpo-$(date +%m%d-%H%M)"

N_VOTES_PER_PROMPT=1
N_SAMPLES_PER_PROMPT=1
BATCH_SIZE=$((N_GPUS * 4))
MICRO_BATCH_SIZE=1
TOTAL_EPOCHS=2
MAX_SAMPLES_PER_EPOCH=1000

MAX_PROMPT_LENGTH=198
MAX_RESPONSE_LENGTH=1024

PRPO_ENABLE=true
PRPO_MIN_PARAGRAPHS=1
PRPO_MAX_PARAGRAPHS=8

ALPHA_CLIP=${ALPHA_CLIP:-0.5}
BETA_CONSISTENCY=${BETA_CONSISTENCY:-0.5}

DEBUG_BATCH_RESPONSES=false
USE_STRUCTURED_RESPONSES=true
DEBUG_PRPO=false

python train/train_prpo_main.py \
    --config-name='ppo_trainer_ttrl_prpo.yaml' \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${BATCH_SIZE} \
    data.filter_overlong_prompts=false \
    data.truncation='error' \
    +data.max_samples_per_epoch=${MAX_SAMPLES_PER_EPOCH} \
    +data.suffix_prompt='' \
    data.tokenizer="${LLAMA_MODEL_PATH}" \
    actor_rollout_ref.model.path="${LLAMA_MODEL_PATH}" \
    critic.model.path="${LLAMA_MODEL_PATH}" \
    critic.model.tokenizer_path="${LLAMA_MODEL_PATH}" \
    reward_model.model.path="${LLAMA_MODEL_PATH}" \
    reward_model.model.input_tokenizer="${LLAMA_MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    +actor_rollout_ref.model.dtype=bfloat16 \
    +critic.model.dtype=bfloat16 \
    +reward_model.model.dtype=bfloat16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${N_GPUS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.optim.warmup_style='cosine' \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.name=functional_embeddings_vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.7 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${N_VOTES_PER_PROMPT} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    algorithm.kl_ctrl.kl_coef=0.00 \
    algorithm.adv_estimator=prpo \
    +use_critic=false \
    custom_reward_function.path="../verl/utils/reward_score/deepfake_detection_prpo_standalone.py" \
    custom_reward_function.name=reward_func \
    ttrl.enable=true \
    ttrl.n_votes_per_prompt=${N_VOTES_PER_PROMPT} \
    ttrl.n_samples_per_prompt=${N_SAMPLES_PER_PROMPT} \
    prpo.enable=${PRPO_ENABLE} \
    prpo.clip_eps=${PRPO_CLIP_EPS} \
    prpo.normalize_adv=${PRPO_NORMALIZE_ADV} \
    prpo.kl_coef=${PRPO_KL_COEF} \
    prpo.entropy_coef=${PRPO_ENTROPY_COEF} \
    prpo.min_paragraphs=${PRPO_MIN_PARAGRAPHS} \
    prpo.max_paragraphs=${PRPO_MAX_PARAGRAPHS} \
    prpo.debug=${DEBUG_PRPO} \
    +prpo.optimized=true \
    trainer.logger=['console','wandb'] \
    trainer.project_name="prpo-training" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=999999 \
    trainer.test_freq=999999 \
    trainer.val_before_train=false \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "$@"
