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
The base tokenizer class, required for any hybrid engine based rollout or inference with vLLM.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
import torch

__all__ = ["HybridEngineBaseTokenizer"]


class HybridEngineBaseTokenizer(ABC):
    """the tokenizer property and function name should align with HF's to meet vllm requirement"""

    @property
    @abstractmethod
    def vocab_size(self):
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        pass

    @property
    @abstractmethod
    def pad_token_id(self):
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        pass

    @property
    @abstractmethod
    def eos_token_id(self):
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        pass

    @property
    @abstractmethod
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        pass

    @property
    @abstractmethod
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).

        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        pass

    @abstractmethod
    def encode(self, text):
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers.

            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers.
        """
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: Union[int, List[int], np.ndarray, torch.Tensor],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        pass

    @abstractmethod
    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        pass

    @abstractmethod
    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
        the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
        something we should change.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        pass

    @abstractmethod
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (`List[str]`): The token to join in a string.

        Returns:
            `str`: The joined tokens.
        """
        pass

    @property
    def is_fast(self):
        return False
