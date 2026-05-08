#!/usr/bin/env python3

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

"""
Custom reward function for deepfake detection task.
Uses the extract_answer_deepfake function to extract answers in format "### Answer: real/fake"
"""

import traceback


def extract_answer_deepfake(response: str) -> str:
    """
    Custom extract_answer function for deepfake detection.
    Extracts 'real' or 'fake' prediction from model response.

    Args:
        response (str): Generated model response

    Returns:
        str: 'real', 'fake', or None if no clear prediction found
    """
    if not isinstance(response, str):
        return None

    prediction = None
    if '### Answer:' in response:
        answer_part = response.split('### Answer:')[-1].strip().lower()
        if 'real' in answer_part and 'fake' not in answer_part:
            prediction = 'real'
        elif 'fake' in answer_part and 'real' not in answer_part:
            prediction = 'fake'
        elif 'fake' in answer_part:  # If both appear, prefer fake (more specific)
            prediction = 'fake'
        elif 'real' in answer_part:
            prediction = 'real'
    else:
        # Fallback: check entire response if no explicit answer section
        response_lower = response.lower()
        if 'fake' in response_lower and 'real' not in response_lower:
            prediction = 'fake'
        elif 'real' in response_lower and 'fake' not in response_lower:
            prediction = 'real'

    return prediction


def compute_score(model_response, gt_answer, fast=True):
    """
    Compute score for deepfake detection task.

    Args:
        model_response (str): Generated model response
        gt_answer (str): Ground truth answer ('real' or 'fake')
        fast (bool): Not used, kept for compatibility

    Returns:
        dict: Score dictionary with format matching TTRL expectations
    """
    model_answer = extract_answer_deepfake(model_response)

    if model_answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": "",
        }

    # Normalize ground truth
    if isinstance(gt_answer, (float, int)):
        gt_answer = str(gt_answer)

    if isinstance(gt_answer, str):
        gt_answer = gt_answer.lower().strip()

    # Check if prediction matches ground truth
    is_correct = (model_answer == gt_answer)

    if is_correct:
        return {
            "score": 1.0,
            "format_score": 1.0,
            "acc": True,
            "extracted_gt": gt_answer,
            "pred": model_answer,
        }
    else:
        return {
            "score": 0.0,
            "format_score": 1.0,  # Give format score if we extracted something
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": model_answer,
        }


def reward_func(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    """
    Main reward function called by TTRL.

    Args:
        data_source: Source of the data (not used)
        solution_str: Generated model response
        ground_truth: Ground truth answer
        extra_info: Additional information (not used)
        sandbox_fusion_url: Sandbox URL (not used)
        concurrent_semaphore: Semaphore for concurrency (not used)

    Returns:
        float or dict: Reward score
    """
    try:
        # Debug ground_truth issue - print occasionally
        # import random
        # if random.randint(1, 200) == 1:  # Print less frequently now that issue is resolved
        #     print("=" * 80)
        #     print("REWARD FUNCTION DEBUG:")
        #     print(f"ground_truth: {ground_truth} (type: {type(ground_truth)})")
        #     print(f"data_source: {data_source}")
        #     print(f"extra_info type: {type(extra_info)}")
        #     if hasattr(extra_info, 'keys'):
        #         print(f"extra_info keys: {extra_info.keys() if extra_info else None}")
        #         print(f"extra_info: {extra_info}")
        #     else:
        #         print(f"extra_info: {extra_info}")
        #     print(f"solution_str: {solution_str[:100]}...")
        #     print("=" * 80)

        if ground_truth is None or str(ground_truth).lower() == 'none':
            print("=" * 80)
            print(f"ERROR: GROUND TRUTH IS NONE-LIKE! Value: '{ground_truth}' (type: {type(ground_truth)})")

            # Try to recover ground truth from extra_info image path
            if extra_info and 'image_path' in extra_info:
                image_path = extra_info['image_path']
                if '/real/' in image_path:
                    ground_truth = 'real'
                    print(f"RECOVERED ground truth as 'real' from path: {image_path}")
                elif '/fake/' in image_path:
                    ground_truth = 'fake'
                    print(f"RECOVERED ground truth as 'fake' from path: {image_path}")
                else:
                    print(f"Could not recover ground truth from path: {image_path}")
                    print("=" * 80)
                    return 0.0
            else:
                print("No extra_info available to recover ground truth")
                print("=" * 80)
                return 0.0
            print("=" * 80)

        res = compute_score(solution_str, str(ground_truth))

        # Print sample responses for debugging (1 in 20 chance - increased frequency)
        # import random
        # if random.randint(1, 20) == 1:
        #     print("=" * 80)
        #     print("🎯 REWARD COMPUTATION SAMPLE:")
        #     print(f"📊 Ground Truth: {ground_truth}")
        #     print(f"🤖 Generated Response: {solution_str[:300]}...")  # First 300 chars
        #     extracted = extract_answer_deepfake(solution_str)
        #     print(f"🔍 Extracted Answer: '{extracted}'")
        #     print(f"🏆 Score Result: {res}")
        #     if isinstance(res, dict):
        #         print(f"   - Final Score: {res.get('score', 0.0)}")
        #         print(f"   - Format Score: {res.get('format_score', 0.0)}")
        #         print(f"   - Accuracy: {res.get('acc', False)}")
        #         print(f"   - Predicted: '{res.get('pred', '')}'")
        #     print("=" * 80)

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])

    except Exception as e:
        print(f"[ERROR] Error in deepfake detection reward function: {str(e)}")
        traceback.print_exc()
        return 0.0
