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
Math utilities for TTRL (originally from TTRL-LLaVA)
Moved to prpo folder to make it self-contained
"""

import re
import string
from typing import Optional, Union


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from a mathematical response.
    Looks for patterns like "The answer is X" or standalone numbers.
    """
    if not text:
        return None

    # Look for "The answer is" pattern
    answer_pattern = r"(?:the answer is|answer:|answer =)\s*([^\n\.,]*)"
    match = re.search(answer_pattern, text.lower())
    if match:
        return match.group(1).strip()

    # Look for numbers at the end
    number_pattern = r"([+-]?\d*\.?\d+)\s*$"
    match = re.search(number_pattern, text.strip())
    if match:
        return match.group(1)

    return None


def simplify_expression_string(expr: str) -> str:
    """
    Simplify a mathematical expression string for comparison.
    Removes whitespace, normalizes formatting.
    """
    if not expr:
        return ""

    # Remove whitespace
    expr = re.sub(r'\s+', '', expr)

    # Normalize decimal points
    expr = re.sub(r'\.0+$', '', expr)

    # Remove leading zeros
    expr = re.sub(r'\b0+(\d)', r'\1', expr)

    return expr.lower()


def grade(ground_truth: str, prediction: str) -> float:
    """
    Grade a mathematical prediction against ground truth.
    Returns 1.0 for correct, 0.0 for incorrect.
    """
    if not ground_truth or not prediction:
        return 0.0

    # Extract and simplify both answers
    gt_answer = extract_answer(ground_truth)
    pred_answer = extract_answer(prediction)

    if not gt_answer or not pred_answer:
        return 0.0

    # Simplify for comparison
    gt_simple = simplify_expression_string(gt_answer)
    pred_simple = simplify_expression_string(pred_answer)

    # Direct string comparison
    if gt_simple == pred_simple:
        return 1.0

    # Try numerical comparison
    try:
        gt_float = float(gt_simple)
        pred_float = float(pred_simple)

        # Allow small floating point errors
        if abs(gt_float - pred_float) < 1e-6:
            return 1.0
    except (ValueError, TypeError):
        pass

    return 0.0


# Additional utility functions that might be used
def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""

    # Remove punctuation and extra whitespace
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def extract_numerical_answer(text: str) -> Optional[float]:
    """Extract a numerical answer from text"""
    if not text:
        return None

    # Look for numbers in the text
    numbers = re.findall(r'[+-]?\d*\.?\d+', text)

    if numbers:
        try:
            return float(numbers[-1])  # Return the last number found
        except ValueError:
            pass

    return None