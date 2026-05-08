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

from typing import List
from collections import Counter
import torch
import numpy as np
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade
from verl.utils.reward_score.deepfake_detection import extract_answer_deepfake


def _compute_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    num_prompts = len(data) // n_votes_per_prompt
    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))
    return data[selected_indices]


def apply_original_gt(batch):
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt
    return batch


def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer):
    num_prompts = len(gen_batch_output) // n

    model_outputs = []
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    majority_gt_list, majority_ratio_list = _batch_majority_vote(model_outputs, n)

    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch


def _batch_majority_vote(model_outputs: List[str], n: int) -> tuple[List[str], List[float]]:
    majority_gt_list = []
    majority_ratio_list = []
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]
        prompt_majority_gt, prompt_majority_ratio = _majority_vote(prompt_outputs)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
    return majority_gt_list, majority_ratio_list


def _majority_vote(model_outputs: List[str]) -> tuple[str, float]:
    model_answers = [extract_answer_deepfake(generated_text) for generated_text in model_outputs]
    valid_answers = [answer for answer in model_answers if answer is not None]
    if len(valid_answers) == 0:
        import random
        fallback_answer = random.choice(['real', 'fake'])
        return fallback_answer, 0.0
    counter = Counter(valid_answers)
    majority_answer, majority_count = counter.most_common(1)[0]
    majority_ratio = majority_count / len(valid_answers)
    return majority_answer, majority_ratio


def compute_ttrl_metrics(batch, n):
    num_prompts = len(batch) // n
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []
    use_paragraph_rewards = ("paragraph_rewards" in batch.non_tensor_batch and
                           "num_paragraphs" in batch.non_tensor_batch)

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        if ("paragraph_rewards" in batch.non_tensor_batch and
            "num_paragraphs" in batch.non_tensor_batch and
            len(batch.non_tensor_batch["paragraph_rewards"]) > idx[i]):
            para_rewards = batch.non_tensor_batch["paragraph_rewards"][idx[i]]
            num_paras = batch.non_tensor_batch["num_paragraphs"][idx[i]]
            if num_paras > 0 and hasattr(para_rewards, '__getitem__'):
                actual_para_rewards = para_rewards[:num_paras]
                if hasattr(actual_para_rewards, 'tolist'):
                    actual_para_rewards = actual_para_rewards.tolist()
                if actual_para_rewards:
                    majority_reward_val = sum(actual_para_rewards) / len(actual_para_rewards)
                else:
                    majority_reward_val = 0.0
            else:
                majority_reward_val = 0.0
            gt_reward_val = majority_reward_val
            majority_reward.append(majority_reward_val)
            gt_reward.append(gt_reward_val)
        else:
            majority_reward.append(0.0)
            gt_reward.append(0.0)

        majority_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"])
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"])

    ttrl_metrics = _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n=n)
    majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
    majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
    ttrl_metrics["majority_ratio"] = majority_ratio

    return ttrl_metrics


def _batch_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: List[str],
    gt_label: List[str],
    n: int,
):
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]
        prompt_majority_label = prompt_majority_label[0]
        prompt_gt_label = prompt_gt_label[0]
        ttrl_metric = _prompt_compute_ttrl_metrics(prompt_majority_reward, prompt_gt_reward, prompt_majority_label, prompt_gt_label)
        ttrl_metrics.append(ttrl_metric)
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}
    return ttrl_metrics

def _prompt_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: str,
    gt_label: str,
    ):
    hit_rate = 1.0 if grade(majority_label, gt_label) else 0.0
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric
