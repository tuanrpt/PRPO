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

import sys
import os
import torch
import numpy as np

from .rl_dataset import RLHFDataset

def _deserialize_tensor(byte_data, tensor_info):
    if tensor_info is None or tensor_info.get("shape") is None:
        return byte_data
    shape = tensor_info["shape"]
    dtype_str = tensor_info["dtype"]
    if hasattr(shape, 'tolist'):
        shape = shape.tolist()
    elif not isinstance(shape, (list, tuple)):
        shape = list(shape) if hasattr(shape, '__iter__') else [shape]
    if "bfloat16" in dtype_str:
        try:
            uint16_array = np.frombuffer(byte_data, dtype=np.uint16).reshape(shape)
            uint16_array = uint16_array.copy()
            tensor = torch.from_numpy(uint16_array).view(torch.bfloat16)
            return tensor
        except Exception as e:
            try:
                float_array = np.frombuffer(byte_data, dtype=np.float32).reshape(shape)
                float_array = float_array.copy()
                tensor = torch.from_numpy(float_array)
                return tensor
            except Exception as e2:
                return torch.zeros(shape, dtype=torch.float32)
    elif "float16" in dtype_str:
        numpy_dtype = np.float16
    elif "float32" in dtype_str:
        numpy_dtype = np.float32
    elif "int64" in dtype_str:
        numpy_dtype = np.int64
    elif "int32" in dtype_str:
        numpy_dtype = np.int32
    else:
        numpy_dtype = np.float32
    try:
        numpy_array = np.frombuffer(byte_data, dtype=numpy_dtype).reshape(shape)
        numpy_array = numpy_array.copy()
        tensor = torch.from_numpy(numpy_array)
        return tensor
    except Exception as e:
        raise e


class EmbeddingsRLHFDataset(RLHFDataset):

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        raw_row_dict = self.dataframe[item]
        has_embeddings = ("inputs_embeds" in raw_row_dict and isinstance(raw_row_dict["inputs_embeds"], bytes) and
                         "attention_mask" in raw_row_dict and isinstance(raw_row_dict["attention_mask"], bytes))
        if has_embeddings:
            self._deserialize_embeddings_to_result(raw_row_dict, row_dict, item)
        return row_dict

    def _deserialize_embeddings_to_result(self, raw_row_dict, row_dict, item):
        try:
            raw_extra_info = raw_row_dict.get("extra_info", {})
            if isinstance(raw_extra_info, str):
                try:
                    import ast
                    raw_extra_info = ast.literal_eval(raw_extra_info)
                except:
                    raw_extra_info = {}
            elif not isinstance(raw_extra_info, dict):
                raw_extra_info = {}
            tensor_info = raw_extra_info.get("tensor_info", {}) if isinstance(raw_extra_info, dict) else {}
            if not tensor_info:
                return
            inputs_embeds_info = tensor_info.get("inputs_embeds", {})
            if not inputs_embeds_info:
                return
            if 'shape' in inputs_embeds_info and hasattr(inputs_embeds_info['shape'], 'tolist'):
                inputs_embeds_info = inputs_embeds_info.copy()
                inputs_embeds_info['shape'] = inputs_embeds_info['shape'].tolist()
            inputs_embeds = _deserialize_tensor(raw_row_dict["inputs_embeds"], inputs_embeds_info)
            if not torch.is_tensor(inputs_embeds):
                return
            attention_mask_info = tensor_info.get("attention_mask", {})
            if not attention_mask_info:
                return
            if 'shape' in attention_mask_info and hasattr(attention_mask_info['shape'], 'tolist'):
                attention_mask_info = attention_mask_info.copy()
                attention_mask_info['shape'] = attention_mask_info['shape'].tolist()
            embedding_attention_mask = _deserialize_tensor(raw_row_dict["attention_mask"], attention_mask_info)
            if not torch.is_tensor(embedding_attention_mask):
                return
            extra_info = row_dict.get("extra_info", {})
            if not isinstance(extra_info, dict):
                extra_info = {}
            extra_info["inputs_embeds"] = inputs_embeds
            extra_info["embedding_attention_mask"] = embedding_attention_mask
            extra_info["has_embeddings"] = True
            row_dict["extra_info"] = extra_info
        except Exception as e:
            raise e
