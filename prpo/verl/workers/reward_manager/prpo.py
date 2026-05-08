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

import re
import torch
from collections import defaultdict
from typing import List, Tuple

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("prpo")
class PRPORewardManager:
    """PRPO-aware reward manager that maps paragraph-level rewards to token-level rewards."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the PRPORewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs using the same logic as PRPO reward function."""
        # Split by double newlines while preserving markdown headers
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean and filter paragraphs
        cleaned = []
        for para in paragraphs:
            para = para.strip()
            
            # Keep the original text with headers - only clean excessive whitespace
            para = re.sub(r'\s+', ' ', para)
            
            # Special handling for final answers - keep them regardless of length
            is_final_answer = bool(re.search(r'###.*[Aa]nswer.*:|[Aa]nswer.*:|[Ff]inal.*[Aa]nswer.*:|[Cc]onclusion.*:', para))
            
            # Keep substantial paragraphs OR final answers (regardless of length)
            if para and (len(para) > 20 or is_final_answer):
                cleaned.append(para)
        
        return cleaned

    def _create_paragraph_token_mapping(self, response_str: str, response_ids: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Create mapping from paragraphs to token ranges.
        
        Args:
            response_str: Decoded response text
            response_ids: Token IDs for the response
            
        Returns:
            List of (start_token_idx, end_token_idx) for each paragraph
        """
        paragraphs = self._split_into_paragraphs(response_str)
        
        if not paragraphs:
            # If no paragraphs found, treat entire response as one paragraph
            return [(0, len(response_ids))]
        
        # Simple approach: divide tokens evenly among paragraphs
        # This is an approximation since exact token-to-text mapping is complex
        total_tokens = len(response_ids)
        tokens_per_paragraph = total_tokens // len(paragraphs)
        remainder_tokens = total_tokens % len(paragraphs)
        
        token_ranges = []
        current_start = 0
        
        for i, paragraph in enumerate(paragraphs):
            # Distribute remainder tokens to first few paragraphs
            current_length = tokens_per_paragraph + (1 if i < remainder_tokens else 0)
            current_end = min(current_start + current_length, total_tokens)
            
            if current_start < total_tokens:
                token_ranges.append((current_start, current_end))
                current_start = current_end
            else:
                # No more tokens left, use empty range
                token_ranges.append((current_start, current_start))
        
        return token_ranges

    def __call__(self, data: DataProto, return_dict=False):
        """
        Compute token-level rewards by mapping paragraph rewards to tokens.
        """
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            reward_model_data = data_item.non_tensor_batch["reward_model"]
            ground_truth = reward_model_data["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {}).copy() if data_item.non_tensor_batch.get("extra_info") else {}
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            
            # Pass through meta_info for validation detection
            if hasattr(data_item, 'meta_info') and data_item.meta_info:
                extra_info["meta_info"] = data_item.meta_info

            # Get reward score from PRPO reward function
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            # Handle PRPO paragraph rewards
            if isinstance(score, dict) and "paragraph_rewards" in score and score.get("prpo_enabled", False):
                paragraph_rewards = score["paragraph_rewards"]
                num_paragraphs = score.get("num_paragraphs", 0)
                
                # Create paragraph to token mapping
                token_ranges = self._create_paragraph_token_mapping(response_str, valid_response_ids)
                
                # Map paragraph rewards to tokens
                actual_paragraphs = min(num_paragraphs, len(paragraph_rewards), len(token_ranges))
                
                for para_idx in range(actual_paragraphs):
                    para_reward = paragraph_rewards[para_idx]
                    
                    # Skip zero-padded paragraphs
                    if para_reward > 0:
                        start_token, end_token = token_ranges[para_idx]
                        
                        # Assign paragraph reward to all tokens in this paragraph
                        for token_idx in range(start_token, end_token):
                            if token_idx < valid_response_length:
                                reward_tensor[i, token_idx] = para_reward
                
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                    
            elif isinstance(score, dict):
                # Handle non-PRPO dict format (fallback to naive behavior)
                if "score" in score:
                    reward = score["score"]
                else:
                    reward = 0.0
                
                # Assign to last token (naive behavior)
                reward_tensor[i, valid_response_length - 1] = reward
                
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                    
            else:
                # Handle scalar reward (fallback to naive behavior)
                reward = score
                reward_tensor[i, valid_response_length - 1] = reward

            # Debug printing
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[PRPO] [prompt]", prompt_str)
                print("[PRPO] [response]", response_str)
                print("[PRPO] [ground_truth]", ground_truth)
                
                if isinstance(score, dict) and "paragraph_rewards" in score:
                    print(f"[PRPO] [paragraph_rewards] {score['paragraph_rewards']}")
                    print(f"[PRPO] [num_paragraphs] {score.get('num_paragraphs', 0)}")
                    
                    # Show token-level reward distribution
                    non_zero_rewards = (reward_tensor[i] != 0).sum().item()
                    print(f"[PRPO] [tokens_with_rewards] {non_zero_rewards}/{valid_response_length}")
                    
                elif isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[PRPO] [{key}]", value)
                else:
                    print("[PRPO] [score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor