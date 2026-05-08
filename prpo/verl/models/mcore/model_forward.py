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

from verl.utils.megatron_utils import unwrap_model

from .util import postprocess_packed_seqs, preprocess_packed_seqs, recover_left_padding, remove_left_padding


def gptmodel_forward(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    """Default forward pass for GPT models with optional sequence packing."""
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )
        if post_process and logits_processor is not None:
            args = {
                k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0]
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_orig, **args)
            output = {
                k: postprocess_packed_seqs(
                    v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                )
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_packed_seqs(
                output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
            )
    else:
        assert logits_processor is None, "logits_processor is not supported for non-packed sequence"
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(
            input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process
        )
        output = model(input_ids=new_input_ids, attention_mask=new_attention_mask, position_ids=new_position_ids)
        output = recover_left_padding(
            output, new_attention_mask, attention_mask, sequence_length, post_process=post_process
        )
    if value_model and post_process:
        output = output[..., 0]
    return output


def gptmodel_forward_qwen2_5_vl(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    multi_modal_inputs=None,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    from megatron.core import parallel_state as mpu

    assert mpu.get_context_parallel_world_size() == 1, "qwen2_5_vl's context parallel is not accurate yet"
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    pixel_values = (
        multi_modal_inputs["pixel_values"].to(input_ids.device) if "pixel_values" in multi_modal_inputs else None
    )
    image_grid_thw = (
        multi_modal_inputs["image_grid_thw"].to(input_ids.device) if "image_grid_thw" in multi_modal_inputs else None
    )
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        if post_process and logits_processor is not None:
            args = {
                k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0]
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_orig, **args)
            output = {
                k: postprocess_packed_seqs(
                    v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                )
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_packed_seqs(
                output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
            )
    else:
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(
            input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process
        )
        output = model(
            input_ids=new_input_ids,
            position_ids=new_position_ids,
            attention_mask=new_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        output = recover_left_padding(
            output, new_attention_mask, attention_mask, sequence_length, post_process=post_process
        )
    if value_model and post_process:
        output = output[..., 0]
    return output
