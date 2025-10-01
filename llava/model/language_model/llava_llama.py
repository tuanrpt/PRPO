# Original Copyright (c) 2023 Haotian Liu (LLaVA)
# Modifications Copyright (c) 2025 Tuan Nguyen
#
# This file is modified from LLaVA: https://github.com/haotian-liu/LLaVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        binary_labels: Optional[torch.FloatTensor] = None,
        lm_loss_weight: Optional[torch.FloatTensor] = None,
        binary_loss_weight: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if return_dict is None:
            return_dict = getattr(self.config, 'use_return_dict', True)

        original_labels = labels

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        if original_labels is not None:
            if labels is not None:
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)

                shift_labels = shift_labels.to(shift_logits.device)
                lm_loss = loss_fct(shift_logits, shift_labels)
            else:
                lm_loss = torch.tensor(0.0, device=outputs.logits.device, dtype=outputs.logits.dtype)

            binary_loss = torch.tensor(0.0, device=lm_loss.device, dtype=lm_loss.dtype)
            binary_logits = None
            
            if binary_labels is not None and images is not None:
                image_features = self.encode_images(images)
                
                binary_logits = self.compute_binary_classification(image_features)
                
                bce_loss_fct = nn.BCEWithLogitsLoss()
                binary_logits_flat = binary_logits.view(-1)
                binary_labels_flat = binary_labels.view(-1).float()
                
                if not torch.all((binary_labels_flat == 0.0) | (binary_labels_flat == 1.0)):
                    print(f"[LOSS WARNING] Invalid binary labels detected! Expected 0 or 1, got: {torch.unique(binary_labels_flat)}")
                    binary_labels_flat = torch.clamp(binary_labels_flat, min=0.0, max=1.0)
                
                binary_logits_flat = torch.clamp(binary_logits_flat, min=-100.0, max=100.0)
                
                try:
                    binary_loss = bce_loss_fct(binary_logits_flat, binary_labels_flat)
                    
                    if torch.isnan(binary_loss).any() or torch.isinf(binary_loss).any():
                        print(f"[LOSS WARNING] NaN/Inf detected in binary_loss, setting to 0.0")
                        binary_loss = torch.tensor(0.0, device=binary_loss.device, dtype=binary_loss.dtype)
                    
                except Exception as e:
                    print(f"[LOSS WARNING] Exception in BCE loss computation: {e}, setting to 0.0")
                    binary_loss = torch.tensor(0.0, device=lm_loss.device, dtype=lm_loss.dtype)

            if lm_loss_weight is not None:
                lm_weight = lm_loss_weight[0].item() if lm_loss_weight.numel() > 0 else 1.0
            else:
                lm_weight = 1.0
                
            if binary_loss_weight is not None:
                binary_weight = binary_loss_weight[0].item() if binary_loss_weight.numel() > 0 else 1.0
            else:
                binary_weight = 1.0
            
            if torch.isnan(lm_loss).any() or torch.isinf(lm_loss).any():
                print(f"[LOSS WARNING] NaN/Inf detected in lm_loss, setting to 0.0")
                lm_loss = torch.tensor(0.0, device=lm_loss.device, dtype=lm_loss.dtype)
                
            if torch.isnan(binary_loss).any() or torch.isinf(binary_loss).any():
                print(f"[LOSS WARNING] NaN/Inf detected in binary_loss, setting to 0.0")
                binary_loss = torch.tensor(0.0, device=binary_loss.device, dtype=binary_loss.dtype)
            
            weighted_lm_loss = lm_weight * lm_loss
            weighted_binary_loss = binary_weight * binary_loss
            total_loss = weighted_lm_loss + weighted_binary_loss
            
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                print(f"[LOSS WARNING] NaN/Inf detected in total_loss, setting to 0.0")
                total_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)

            outputs.loss = total_loss

            outputs.lm_loss = lm_loss
            outputs.binary_loss = binary_loss
            outputs.lm_loss_weight = lm_weight
            outputs.binary_loss_weight = binary_weight
            outputs.weighted_lm_loss = weighted_lm_loss
            outputs.weighted_binary_loss = weighted_binary_loss
            
            if binary_logits is not None:
                outputs.binary_logits = binary_logits

        if not return_dict:
            output = (outputs.logits,) + outputs[1:]
            if original_labels is not None and hasattr(outputs, 'loss'):
                return (outputs.loss,) + output
            else:
                return output

        return outputs

    def forward_binary_only(
        self,
        images: Optional[torch.FloatTensor] = None,
        binary_labels: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for binary classification only using augmented images.
        This bypasses the language model and only processes images through vision encoder and binary classifier.
        
        Args:
            images: Augmented images for binary classification
            binary_labels: Binary classification targets
            return_dict: Whether to return a ModelOutput object
            
        Returns:
            Output with binary_loss and binary_logits
        """
        if return_dict is None:
            return_dict = getattr(self.config, 'use_return_dict', True)

        binary_loss = torch.tensor(0.0, device=images.device, dtype=images.dtype)
        binary_logits = None
        
        if binary_labels is not None and images is not None:
            image_features = self.encode_images(images)
            
            binary_logits = self.compute_binary_classification(image_features)
            
            bce_loss_fct = nn.BCEWithLogitsLoss()
            binary_logits_flat = binary_logits.view(-1)
            binary_labels_flat = binary_labels.view(-1).float()
            binary_loss = bce_loss_fct(binary_logits_flat, binary_labels_flat)

        class BinaryOnlyOutput:
            def __init__(self, loss, binary_logits):
                self.loss = loss
                self.binary_loss = loss
                self.binary_logits = binary_logits
                self.logits = None
                self.past_key_values = None
                self.hidden_states = None
                self.attentions = None

        outputs = BinaryOnlyOutput(binary_loss, binary_logits)
        
        if not return_dict:
            output = (binary_logits,) if binary_logits is not None else ()
            if binary_labels is not None:
                return (binary_loss,) + output
            else:
                return output

        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
