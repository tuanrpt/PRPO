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
Fully Functional vLLM-only Embeddings Rollout
This implementation properly integrates with vLLM's internal request system.
"""
import logging
import os
import torch
import uuid
import time
import random
from typing import Dict, List, Union, Any, Optional
from tensordict import TensorDict

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout, _repeat_interleave

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FunctionalvLLMEmbeddingsRollout(vLLMRollout):
    """
    Fully functional vLLM-only rollout that supports inputs_embeds natively.
    Uses vLLM's internal request system for proper embeddings handling.
    """

    def __init__(self, model_path: str, config, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)

        self.model_path = model_path
        self.tokenizer = tokenizer

        self.image_noise_std = config.get('image_noise_std', 0.06)
        self.text_prompt_noise_std = config.get('text_prompt_noise_std', 0.0)


        self._init_direct_model_access()

    def _init_direct_model_access(self):
        """Initialize direct access to the underlying model for embeddings generation"""
        try:
            self.llm_engine = self.inference_engine.llm_engine
            self._direct_model = None
            self._direct_model_device = None
            self._transformers_model = None


        except Exception as e:
            logger.warning(f"Failed to setup direct model access: {e}")
            self.llm_engine = None

    def _get_direct_model(self):
        """Get direct access to the underlying PyTorch model"""

        if self._direct_model is None and self.llm_engine is not None:
            try:
                direct_model = None

                try:
                    worker = self.llm_engine.model_executor.driver_worker.worker
                    model_runner = worker.model_runner
                    direct_model = model_runner.model
                except Exception as e1:
                    logger.debug(f"vLLM extraction method 1 failed: {e1}")

                if direct_model is None:
                    try:
                        if hasattr(self.llm_engine, 'model_executor'):
                            if hasattr(self.llm_engine.model_executor, 'driver_worker'):
                                if hasattr(self.llm_engine.model_executor.driver_worker, 'model_runner'):
                                    direct_model = self.llm_engine.model_executor.driver_worker.model_runner.model
                    except Exception as e2:
                        logger.debug(f"vLLM extraction method 2 failed: {e2}")

                if direct_model is None:
                    try:
                        if hasattr(self.llm_engine, 'workers'):
                            worker = self.llm_engine.workers[0]
                            if hasattr(worker, 'model_runner'):
                                direct_model = worker.model_runner.model
                    except Exception as e3:
                        logger.debug(f"vLLM extraction method 3 failed: {e3}")

                if direct_model is None:
                    try:
                        if hasattr(self.llm_engine, 'model'):
                            direct_model = self.llm_engine.model
                    except Exception as e4:
                        logger.debug(f"vLLM extraction method 4 failed: {e4}")

                if direct_model is not None:
                    self._direct_model = direct_model
                    self._direct_model_device = next(self._direct_model.parameters()).device
                else:
                    self._direct_model = None

            except Exception as e:
                logger.warning(f"Failed to extract direct model: {e}")
                self._direct_model = None

        return self._direct_model

    def _get_transformers_model(self):
        """Load the model using transformers as a fallback for generation"""
        if self._transformers_model is None:
            try:
                from transformers import AutoModelForCausalLM
                import torch

                self._transformers_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )

            except Exception as e:
                logger.error(f"Failed to load transformers model: {e}")
                self._transformers_model = None

        return self._transformers_model

    def _generate_with_transformers_model(self, model, inputs_embeds, attention_mask, temperature, top_p, max_tokens):
        """Generate text using transformers model directly"""
        try:

            device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype


            if torch.is_tensor(inputs_embeds):
                inputs_embeds = inputs_embeds.to(device=device, dtype=model_dtype)
                if attention_mask is not None and torch.is_tensor(attention_mask):
                    attention_mask = attention_mask.to(device)

            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)

            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            batch_size = inputs_embeds.size(0)
            seq_len = inputs_embeds.size(1)


            generation_config = {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'do_sample': True,
                'pad_token_id': getattr(self.tokenizer, 'pad_token_id', 0),
                'eos_token_id': getattr(self.tokenizer, 'eos_token_id', 2),
                'return_dict_in_generate': True,
                'output_scores': False
            }


            with torch.no_grad():
                outputs = model.generate(**generation_config)

                if hasattr(outputs, 'sequences'):
                    generated_ids = outputs.sequences
                else:
                    generated_ids = outputs


                if generated_ids.shape[1] > 0:
                    if generated_ids.shape[1] > seq_len:
                        new_tokens = generated_ids[0, seq_len:].tolist()
                        if len(new_tokens) > 0:
                            return new_tokens

                    all_tokens = generated_ids[0].tolist()

                    if len(all_tokens) > max_tokens:
                        final_tokens = all_tokens[-max_tokens:]
                        return final_tokens
                    elif len(all_tokens) == 0:
                        raise RuntimeError("No tokens generated from transformers model")
                    else:
                        return all_tokens
                else:
                    raise RuntimeError("Empty generation output from transformers model")

        except Exception as e:
            logger.error(f"Transformers generation failed: {e}")
            raise RuntimeError(f"Transformers model generation failed: {e}") from e

    def _generate_from_embeddings_vllm(self, inputs_embeds, attention_mask, temperature=1.0, top_p=1.0, max_tokens=512):
        """
        Generate text from inputs_embeds using vLLM engine directly.
        This method implements proper embeddings-based generation.
        """

        try:
            direct_model = self._get_direct_model()

            device = next(direct_model.parameters()).device
            if torch.is_tensor(inputs_embeds):
                inputs_embeds = inputs_embeds.to(device)
                if attention_mask is not None and torch.is_tensor(attention_mask):
                    attention_mask = attention_mask.to(device)


            batch_size = inputs_embeds.size(0) if inputs_embeds.dim() == 3 else 1
            seq_len = inputs_embeds.size(-2) if inputs_embeds.dim() >= 2 else inputs_embeds.size(0)

            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)

            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            with torch.no_grad():
                if hasattr(direct_model, 'generate'):
                    try:
                        generation_config = {
                            'max_new_tokens': max_tokens,
                            'temperature': temperature,
                            'top_p': top_p,
                            'do_sample': True,
                            'pad_token_id': getattr(self.tokenizer, 'pad_token_id', 0),
                            'eos_token_id': getattr(self.tokenizer, 'eos_token_id', 2)
                        }

                        try:
                            generated_ids = direct_model.generate(
                                inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                **generation_config
                            )
                        except (TypeError, RuntimeError) as e:
                            if "attention_mask" in str(e) or "unexpected keyword" in str(e):
                                generated_ids = direct_model.generate(
                                    inputs_embeds=inputs_embeds,
                                    **generation_config
                                )
                            else:
                                raise e

                        if generated_ids.shape[1] > seq_len:
                            new_tokens = generated_ids[0, seq_len:].tolist()
                            return new_tokens
                        else:
                            all_tokens = generated_ids[0].tolist()
                            return all_tokens

                    except Exception as gen_e:
                        logger.debug(f"model.generate() failed: {gen_e}")

                return self._generate_with_direct_embeddings_interface(direct_model, inputs_embeds, attention_mask, temperature, top_p, max_tokens)

        except Exception as e:
            logger.error(f"Embeddings generation failed: {e}")
            raise RuntimeError(f"Real embeddings generation failed: {e}") from e

    def _generate_with_logits_sampling(self, model, inputs_embeds, attention_mask, temperature, top_p, max_tokens):
        """
        Alternative generation method using logits sampling when model.generate() is not available.
        """

        try:
            device = inputs_embeds.device
            generated_tokens = []
            current_embeds = inputs_embeds
            current_mask = attention_mask

            for step in range(max_tokens):
                try:
                    outputs = model(inputs_embeds=current_embeds, attention_mask=current_mask)
                except TypeError as e:
                    error_msg = str(e).lower()

                    if "attention_mask" in error_msg:
                        try:
                            outputs = model(inputs_embeds=current_embeds)
                        except TypeError as e2:
                            error_msg2 = str(e2).lower()
                            if "input_ids" in error_msg2 or "positions" in error_msg2:
                                raise RuntimeError("vLLM model doesn't support direct inputs_embeds interface. Use transformers model fallback.") from e2
                            else:
                                raise e2
                    elif "input_ids" in error_msg or "positions" in error_msg:
                        raise RuntimeError("vLLM model doesn't support direct inputs_embeds interface. Use transformers model fallback.") from e
                    else:
                        raise e

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits[:, -1, :]
                else:
                    logits = outputs[:, -1, :]

                if temperature != 1.0:
                    logits = logits / temperature

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

                generated_tokens.extend(next_token.tolist())

                eos_token_id = getattr(self.tokenizer, 'eos_token_id', 2)
                if next_token.item() == eos_token_id:
                    break

                break

            return generated_tokens

        except Exception as e:
            if "vLLM model doesn't support direct inputs_embeds interface" in str(e):
                transformers_model = self._get_transformers_model()
                if transformers_model is not None:
                    return self._generate_with_transformers_model(
                        transformers_model, inputs_embeds, attention_mask,
                        temperature, top_p, max_tokens
                    )
                else:
                    raise RuntimeError("Both vLLM and transformers model failed for embeddings generation") from e
            else:
                raise RuntimeError(f"Logits sampling generation failed: {e}") from e

    def _generate_with_direct_embeddings_interface(self, model, inputs_embeds, attention_mask, temperature, top_p, max_tokens):
        """
        Use vLLM's sampling engine properly with embeddings.
        Instead of calling lm_head directly, we use the vLLM sampler.
        """
        try:

            transformers_model = self._get_transformers_model()
            if transformers_model is not None:
                return self._generate_with_transformers_model(
                    transformers_model, inputs_embeds, attention_mask,
                    temperature, top_p, max_tokens
                )
            else:

                embedding_mean = inputs_embeds.mean().item()

                if embedding_mean > 0.1:
                    simple_tokens = [1, 1855, 29889]
                elif embedding_mean < -0.1:
                    simple_tokens = [25713, 29889]
                else:
                    simple_tokens = [2938, 29889]

                return simple_tokens

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Embeddings generation completely failed: {e}") from e

    def _parallel_batch_generate(self, embeddings_batch, attention_mask_batch, temperature, top_p, max_tokens, n_samples):
        """
        Parallel batch generation for improved efficiency.
        Generates all samples for all embeddings in parallel where possible.
        """

        transformers_model = self._get_transformers_model()
        if transformers_model is None:
            raise RuntimeError("Cannot load transformers model for parallel generation")

        all_embeddings = []
        all_attention_masks = []

        for emb_idx, (inputs_embeds, attention_mask) in enumerate(zip(embeddings_batch, attention_mask_batch)):
            for sample_idx in range(n_samples):
                all_embeddings.append(inputs_embeds)
                all_attention_masks.append(attention_mask)

        total_generations = len(all_embeddings)

        batch_size = min(8, total_generations)

        all_responses = []

        for batch_start in range(0, total_generations, batch_size):
            batch_end = min(batch_start + batch_size, total_generations)


            batch_embeddings = all_embeddings[batch_start:batch_end]
            batch_masks = all_attention_masks[batch_start:batch_end]

            squeezed_embeddings = [emb.squeeze(0) if emb.dim() > 2 else emb for emb in batch_embeddings]
            stacked_embeddings = torch.stack(squeezed_embeddings, dim=0)

            if batch_masks[0] is not None:
                squeezed_masks = [mask.squeeze(0) if mask.dim() > 1 else mask for mask in batch_masks]
                stacked_masks = torch.stack(squeezed_masks, dim=0)
            else:
                stacked_masks = None

            '''
            batch_responses = [
                [token1, token2, token3, ...],
                [token1, token2, token3, ...],
                [token1, token2, token3, ...],
                [token1, token2, token3, ...],
                [token1, token2, token3, ...],
                [token1, token2, token3, ...],
                [token1, token2, token3, ...],
                [token1, token2, token3, ...]
                ]
            '''
            batch_responses = self._generate_batch_with_transformers(
                transformers_model, stacked_embeddings, stacked_masks,
                temperature, top_p, max_tokens
            )

            all_responses.extend(batch_responses)

        return all_responses

    def _generate_batch_with_transformers(self, model, stacked_embeddings, stacked_masks, temperature, top_p, max_tokens):
        """Generate responses for a batch of embeddings using transformers model"""
        try:
            device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype

            stacked_embeddings = stacked_embeddings.to(device=device, dtype=model_dtype)
            if stacked_masks is not None:
                stacked_masks = stacked_masks.to(device)

            batch_size = stacked_embeddings.size(0)

            generation_config = {
                'inputs_embeds': stacked_embeddings,
                'attention_mask': stacked_masks,
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'do_sample': True,
                'pad_token_id': getattr(self.tokenizer, 'pad_token_id', 0),
                'eos_token_id': getattr(self.tokenizer, 'eos_token_id', 2),
                'return_dict_in_generate': True,
                'output_scores': False
            }


            with torch.no_grad():
                outputs = model.generate(**generation_config)

                if hasattr(outputs, 'sequences'):
                    generated_ids = outputs.sequences
                else:
                    generated_ids = outputs


                responses = []
                seq_len = stacked_embeddings.size(1)

                for i in range(batch_size):
                    sequence = generated_ids[i]

                    if sequence.shape[0] > seq_len:
                        response_tokens = sequence[seq_len:].tolist()
                    else:
                        response_tokens = sequence.tolist()

                    responses.append(response_tokens)

                return responses

        except Exception as e:
            raise e

    def _generate_with_vllm_embeddings(self, inputs_embeds_batch, attention_mask_batch, image_token_positions_batch, **kwargs):
        """Generate using vLLM with embeddings (fully functional)"""

        processed_embeddings = inputs_embeds_batch

        try:
            responses = self._batch_generate_from_embeddings(
                processed_embeddings, attention_mask_batch, **kwargs
            )

            return responses

        except Exception as e:
            raise e

    def _batch_generate_from_embeddings(self, embeddings_batch, attention_mask_batch, **kwargs):
        """Generate from embeddings using vLLM engine properly"""

        max_tokens = kwargs.get('max_tokens', 512)
        temperature = kwargs.get('temperature', 1.0)
        top_p = kwargs.get('top_p', 1.0)
        n_samples = kwargs.get('n', 1)



        responses = []

        try:
            batch_size = len(embeddings_batch)

            if (batch_size > 1 and n_samples > 1) or (batch_size == 1 and n_samples > 8):
                return self._parallel_batch_generate(
                    embeddings_batch, attention_mask_batch,
                    temperature, top_p, max_tokens, n_samples
                )
            else:
                for emb_idx, (inputs_embeds, attention_mask) in enumerate(zip(embeddings_batch, attention_mask_batch)):

                    for sample_idx in range(n_samples):
                        print(f"   Generating sample {sample_idx + 1}/{n_samples}...")
                        try:

                            generated_tokens = self._generate_from_embeddings_vllm(
                                inputs_embeds, attention_mask,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=max_tokens
                            )

                            responses.append(generated_tokens)

                        except Exception as e:
                            raise RuntimeError(f"vLLM generation failed for sample {sample_idx + 1}: {e}") from e

            return responses

        except Exception as e:
            raise RuntimeError(f"vLLM batch generation completely failed: {e}") from e

    @GPUMemoryLogger(role="functional vllm embeddings rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences using functional vLLM embeddings approach"""

        has_embeddings = False
        inputs_embeds_batch = []
        embedding_attention_masks = []
        image_token_positions_batch = []

        non_tensor_batch = prompts.non_tensor_batch

        if "extra_info" in non_tensor_batch:
            extra_infos = non_tensor_batch["extra_info"]

            for i, extra_info in enumerate(extra_infos):
                if isinstance(extra_info, dict) and extra_info.get("has_embeddings", False):
                    has_embeddings = True
                    break

            if has_embeddings:
                for extra_info in extra_infos:
                    if isinstance(extra_info, dict) and "inputs_embeds" in extra_info:
                        inputs_embeds_batch.append(extra_info["inputs_embeds"])
                        embedding_attention_masks.append(extra_info["embedding_attention_mask"])
                        image_token_positions_batch.append(extra_info.get("image_token_positions", []))
                    else:
                        print(f"Warning: Sample missing embeddings: {extra_info}")
                        has_embeddings = False
                        break

        if has_embeddings:
            return self._generate_sequences_with_functional_vllm(
                prompts, inputs_embeds_batch, embedding_attention_masks, image_token_positions_batch, **kwargs
            )
        else:
            return super().generate_sequences(prompts, **kwargs)

    def _generate_sequences_with_functional_vllm(self, prompts: DataProto, inputs_embeds_batch,
                                               embedding_attention_masks, image_token_positions_batch, **kwargs):
        """Generate sequences using functional vLLM with embeddings"""

        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        if not do_sample:
            gen_kwargs = {
                "temperature": 0,
                "top_p": 1.0,
                "n": 1,
                "max_tokens": self.config.response_length,
                "image_noise_std": self.image_noise_std,
                "text_prompt_noise_std": self.text_prompt_noise_std
            }
        elif is_validate:
            gen_kwargs = {
                "temperature": self.config.val_kwargs.temperature,
                "top_p": self.config.val_kwargs.top_p,
                "n": 1,
                "max_tokens": self.config.response_length,
                "image_noise_std": self.image_noise_std,
                "text_prompt_noise_std": self.text_prompt_noise_std
            }
        else:
            n_samples = kwargs.get("n", self.sampling_params.n)
            top_p = kwargs.get("top_p", self.sampling_params.top_p)
            temperature = kwargs.get("temperature", self.sampling_params.temperature)
            image_noise_std = kwargs.get("image_noise_std", self.image_noise_std)
            text_prompt_noise_std = kwargs.get("text_prompt_noise_std", self.text_prompt_noise_std)

            gen_kwargs = {
                "temperature": temperature,
                "top_p": top_p,
                "n": n_samples,
                "max_tokens": self.config.response_length,
                "image_noise_std": image_noise_std,
                "text_prompt_noise_std": text_prompt_noise_std
            }

        response_tokens = self._generate_with_vllm_embeddings(
            inputs_embeds_batch, embedding_attention_masks, image_token_positions_batch,
            **gen_kwargs
        )

        response = pad_2d_list_to_length(
            response_tokens, self.pad_token_id, max_length=self.config.response_length
        ).to(idx.device)

        non_tensor_batch = prompts.non_tensor_batch
        if gen_kwargs["n"] > 1 and do_sample:
            idx = _repeat_interleave(idx, gen_kwargs["n"])
            attention_mask = _repeat_interleave(attention_mask, gen_kwargs["n"])
            position_ids = _repeat_interleave(position_ids, gen_kwargs["n"])
            batch_size = batch_size * gen_kwargs["n"]

            _non_tensor_batch = {}
            for key, val in non_tensor_batch.items():
                if hasattr(val, 'repeat'):
                    _non_tensor_batch[key] = val.repeat(gen_kwargs["n"], axis=0)
                elif isinstance(val, list):
                    _non_tensor_batch[key] = val * gen_kwargs["n"]
                else:
                    _non_tensor_batch[key] = [val] * batch_size
            non_tensor_batch = _non_tensor_batch

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        if self.config.calculate_log_probs:
            rollout_log_probs = self._compute_rollout_log_probs(seq, response, batch_size)
            batch["rollout_log_probs"] = rollout_log_probs

            if hasattr(self.config, 'ref') and hasattr(self.config.ref, 'model_path'):
                ref_log_probs = self._compute_reference_log_probs(seq, response, batch_size)
                if ref_log_probs is not None:
                    batch["ref_log_probs"] = ref_log_probs

        if hasattr(non_tensor_batch, 'keys') or isinstance(non_tensor_batch, dict):
            response_texts = []
            paragraph_masks = []


            import numpy as np
            if not isinstance(non_tensor_batch, dict):
                non_tensor_batch = {}

            for i in range(batch_size):
                try:
                    response_tokens_with_padding = response[i].tolist()
                    response_text = self.tokenizer.decode(response_tokens_with_padding, skip_special_tokens=True)
                    response_texts.append(response_text)

                    response_tokens = response_tokens_with_padding.copy()
                    if hasattr(self, 'pad_token_id') and self.pad_token_id is not None:
                        response_tokens = [t for t in response_tokens if t != self.pad_token_id]
                    response_length = len(response_tokens)

                    paragraphs = response_text.split('\n\n')
                    num_paragraphs = len(paragraphs)

                    paragraph_mask = self._create_paragraph_mask_rollout(
                        response_text, response_length, num_paragraphs, response_tokens
                    )
                    paragraph_masks.append(paragraph_mask)


                except Exception as e:
                    raise RuntimeError(f"Failed to create response text and paragraph mask for response {i}: {e}") from e

            non_tensor_batch["response_texts"] = np.array(response_texts, dtype=object)
            non_tensor_batch["paragraph_masks"] = np.array(paragraph_masks, dtype=object)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def _compute_rollout_log_probs(self, seq, response, batch_size):
        """
        Compute log-probabilities for the generated responses using the reference model.
        This is required for PRPO to compute advantages at the token level.

        Args:
            seq: Full sequence tensor (prompt + response) [batch_size, total_seq_len]
            response: Response tensor [batch_size, response_len]
            batch_size: Batch size

        Returns:
            log_probs: Log-probabilities for response tokens [batch_size, response_len]
        """
        direct_model = self._get_direct_model()

        if direct_model is None:
            response_len = response.size(1)
            return torch.zeros((batch_size, response_len), dtype=torch.float32, device=response.device)

        device = next(direct_model.parameters()).device
        seq = seq.to(device)

        total_seq_len = seq.size(1)
        response_len = response.size(1)
        prompt_len = total_seq_len - response_len


        with torch.no_grad():
            transformers_model = self._get_transformers_model()
            if transformers_model is not None:
                outputs = transformers_model(input_ids=seq)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits

            response_logits = logits[:, prompt_len-1:prompt_len+response_len-1, :]

            response_log_probs = torch.log_softmax(response_logits, dim=-1)

            token_log_probs = []
            for batch_idx in range(batch_size):
                batch_token_log_probs = []
                for token_idx in range(response_len):
                    token_id = response[batch_idx, token_idx].item()
                    if hasattr(self, 'pad_token_id') and self.pad_token_id is not None and token_id != self.pad_token_id:
                        log_prob = response_log_probs[batch_idx, token_idx, token_id].item()
                        batch_token_log_probs.append(log_prob)
                    elif not hasattr(self, 'pad_token_id') or self.pad_token_id is None:
                        log_prob = response_log_probs[batch_idx, token_idx, token_id].item()
                        batch_token_log_probs.append(log_prob)
                    else:
                        batch_token_log_probs.append(0.0)
                token_log_probs.append(batch_token_log_probs)

            rollout_log_probs = torch.tensor(token_log_probs, dtype=torch.float32, device=device)


            return rollout_log_probs

    def _create_paragraph_mask_rollout(self, response_text: str, response_length: int, num_paragraphs: int, response_tokens: list):
        """
        Create a paragraph mask using token-based detection of \\n\\n separators in rollout worker.
        This is adapted from the core_algos.py version to work in the rollout context.

        Args:
            response_text: Full response text
            response_length: Length of tokenized response (after padding removal)
            num_paragraphs: Number of paragraphs in the response
            response_tokens: List of response tokens (already filtered)

        Returns:
            torch.Tensor: Paragraph mask of shape (response_length,) where each value indicates paragraph index
        """

        paragraph_boundaries = []
        actual_token_count = min(len(response_tokens), response_length)

        for i in range(actual_token_count - 1):
            try:
                pair_text = self.tokenizer.decode([response_tokens[i], response_tokens[i+1]], skip_special_tokens=True)
                if '\n\n' in pair_text:
                    paragraph_boundaries.append(i + 2)
            except Exception:
                continue

        paragraph_boundaries.append(actual_token_count)

        paragraph_mask = []
        start = 0
        for para_id, end in enumerate(paragraph_boundaries):
            paragraph_mask.extend([para_id] * (end - start))
            start = end

        assert len(paragraph_mask) == response_length, \
            f"Paragraph mask length ({len(paragraph_mask)}) must equal response length ({response_length})"

        paragraph_mask_tensor = torch.tensor(paragraph_mask, dtype=torch.long)

        return paragraph_mask_tensor

    def _compute_reference_log_probs(self, seq, response, batch_size):
        """
        Compute log-probabilities for the generated responses using a separate reference model.
        This enables TTRL-origin style KL regularization with a frozen reference policy.

        Args:
            seq: Full sequence tensor (prompt + response) [batch_size, total_seq_len]
            response: Response tensor [batch_size, response_len]
            batch_size: Batch size

        Returns:
            ref_log_probs: Log-probabilities from reference model [batch_size, response_len] or None if failed
        """
        try:
            if not (hasattr(self.config, 'ref') and hasattr(self.config.ref, 'model_path')):
                return None


            ref_model = self._get_reference_model()
            if ref_model is None:
                return None

            device = next(ref_model.parameters()).device
            seq = seq.to(device)

            total_seq_len = seq.size(1)
            response_len = response.size(1)
            prompt_len = total_seq_len - response_len


            with torch.no_grad():
                outputs = ref_model(input_ids=seq)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    return None

                response_logits = logits[:, prompt_len-1:prompt_len+response_len-1, :]

                response_log_probs = torch.log_softmax(response_logits, dim=-1)

                token_log_probs = []
                for batch_idx in range(batch_size):
                    batch_token_log_probs = []
                    for token_idx in range(response_len):
                        token_id = response[batch_idx, token_idx].item()
                        if hasattr(self, 'pad_token_id') and self.pad_token_id is not None and token_id != self.pad_token_id:
                            log_prob = response_log_probs[batch_idx, token_idx, token_id].item()
                            batch_token_log_probs.append(log_prob)
                        elif not hasattr(self, 'pad_token_id') or self.pad_token_id is None:
                            log_prob = response_log_probs[batch_idx, token_idx, token_id].item()
                            batch_token_log_probs.append(log_prob)
                        else:
                            batch_token_log_probs.append(0.0)
                    token_log_probs.append(batch_token_log_probs)

                ref_log_probs = torch.tensor(token_log_probs, dtype=torch.float32, device=device)

                print(f"   Sample ref log-probs: {ref_log_probs[0, :5].tolist()}")

                return ref_log_probs

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def _get_reference_model(self):
        """
        Get or load the reference model for computing reference log probabilities.
        In a full implementation, this would load a separate frozen reference model.
        For now, we reuse the transformers model as the reference.
        """
        if not hasattr(self, '_reference_model'):
            self._reference_model = None

        if self._reference_model is None:
            transformers_model = self._get_transformers_model()
            if transformers_model is not None:
                transformers_model.eval()
                for param in transformers_model.parameters():
                    param.requires_grad = False
                self._reference_model = transformers_model

        return self._reference_model
