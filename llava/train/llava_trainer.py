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

import os
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

try:
    from peft import PeftModel, is_peft_available
except ImportError:
    def is_peft_available():
        return False
    PeftModel = None

try:
    from apex import amp
except ImportError:
    amp = None
from typing import List, Optional, Dict, Any


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_predictions = []
        self.binary_targets = []
        self.binary_eval_step = 0

    def training_step(self, model, inputs):
        """Override training_step to log individual loss components"""
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            from transformers.trainer import smp_forward_backward
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if hasattr(outputs, "lm_loss") and hasattr(outputs, "binary_loss"):
            lm_loss_val = outputs.lm_loss.detach().mean()
            binary_loss_val = outputs.binary_loss.detach().mean()

            log_dict = {
                "loss": round(loss.detach().item(), 3),
                "lm_loss": round(lm_loss_val.item() if hasattr(lm_loss_val, 'item') else lm_loss_val, 3),
                "binary_loss": round(binary_loss_val.item() if hasattr(binary_loss_val, 'item') else binary_loss_val, 3)
            }

            if hasattr(outputs, "lm_loss_weight"):
                log_dict["lm_loss_weight"] = float(outputs.lm_loss_weight)
            if hasattr(outputs, "binary_loss_weight"):
                log_dict["binary_loss_weight"] = float(outputs.binary_loss_weight)

            self.log(log_dict)

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps


    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to extract and log individual loss components"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        has_binary_images = 'images_binary' in inputs
        binary_images = inputs.pop('images_binary', None)

        if has_binary_images and 'binary_labels' in inputs:
            lm_inputs = inputs.copy()
            lm_outputs = model(**lm_inputs)

            binary_inputs = {
                'images': binary_images,
                'binary_labels': inputs['binary_labels'],
                'return_dict': True
            }

            binary_outputs = model.forward_binary_only(**binary_inputs)

            if hasattr(lm_outputs, 'loss'):
                lm_loss = lm_outputs.loss
            elif isinstance(lm_outputs, dict) and 'loss' in lm_outputs:
                lm_loss = lm_outputs['loss']
            else:
                lm_loss = lm_outputs[0] if isinstance(lm_outputs, (list, tuple)) else lm_outputs

            if hasattr(binary_outputs, 'loss'):
                binary_loss = binary_outputs.loss
            elif isinstance(binary_outputs, dict) and 'loss' in binary_outputs:
                binary_loss = binary_outputs['loss']
            else:
                binary_loss = binary_outputs[0] if isinstance(binary_outputs, (list, tuple)) else binary_outputs

            lm_weight = 1.0
            binary_weight = 1.0
            if 'lm_loss_weight' in inputs:
                lm_weight = inputs['lm_loss_weight'][0].item() if inputs['lm_loss_weight'].numel() > 0 else 1.0
            if 'binary_loss_weight' in inputs:
                binary_weight = inputs['binary_loss_weight'][0].item() if inputs['binary_loss_weight'].numel() > 0 else 1.0

            total_loss = lm_weight * lm_loss + binary_weight * binary_loss

            class CombinedOutputs:
                def __init__(self, lm_loss, binary_loss, total_loss, lm_outputs, lm_weight, binary_weight):
                    self.lm_loss = lm_loss
                    self.binary_loss = binary_loss
                    self.loss = total_loss
                    self.lm_loss_weight = lm_weight
                    self.binary_loss_weight = binary_weight
                    self.weighted_lm_loss = lm_weight * lm_loss
                    self.weighted_binary_loss = binary_weight * binary_loss
                    self.logits = getattr(lm_outputs, 'logits', None)
                    if hasattr(lm_outputs, 'past_key_values'):
                        self.past_key_values = lm_outputs.past_key_values
                    if hasattr(lm_outputs, 'hidden_states'):
                        self.hidden_states = lm_outputs.hidden_states
                    if hasattr(lm_outputs, 'attentions'):
                        self.attentions = lm_outputs.attentions

            outputs = CombinedOutputs(lm_loss, binary_loss, total_loss, lm_outputs, lm_weight, binary_weight)
        else:
            outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            elif isinstance(outputs, dict):
                if "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                loss = outputs["loss"]
            else:
                loss = outputs[0]

        if hasattr(outputs, 'lm_loss') and hasattr(outputs, 'binary_loss'):
            lm_loss = outputs.lm_loss.detach().item() if hasattr(outputs.lm_loss, 'item') else float(outputs.lm_loss)
            binary_loss = outputs.binary_loss.detach().item() if hasattr(outputs.binary_loss, 'item') else float(outputs.binary_loss)
            total_loss = loss.detach().item() if hasattr(loss, 'item') else float(loss)

            self._current_lm_loss = round(lm_loss, 3)
            self._current_binary_loss = round(binary_loss, 3)
            self._current_total_loss = round(total_loss, 3)


        return (loss, outputs) if return_outputs else loss




    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluation loop to log individual loss components and binary classification metrics during evaluation"""
        prediction_loss_only = getattr(self.args, 'prediction_loss_only', False)

        eval_result = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        if hasattr(self, '_eval_lm_losses') and hasattr(self, '_eval_binary_losses'):
            avg_lm_loss = np.mean(self._eval_lm_losses) if self._eval_lm_losses else 0.0
            avg_binary_loss = np.mean(self._eval_binary_losses) if self._eval_binary_losses else 0.0

            if not hasattr(eval_result, 'metrics'):
                eval_result.metrics = {}
            eval_result.metrics[f"{metric_key_prefix}_lm_loss"] = round(avg_lm_loss, 3)
            eval_result.metrics[f"{metric_key_prefix}_binary_loss"] = round(avg_binary_loss, 3)

            if WANDB_AVAILABLE and wandb.run is not None:
                wandb_metrics = {
                    f"{metric_key_prefix}/lm_loss": round(avg_lm_loss, 3),
                    f"{metric_key_prefix}/binary_loss": round(avg_binary_loss, 3)
                }
                wandb.log(wandb_metrics)

            if hasattr(self, '_eval_binary_preds') and hasattr(self, '_eval_binary_targets'):
                self._eval_binary_preds = []
                self._eval_binary_targets = []

            self._eval_lm_losses = []
            self._eval_binary_losses = []

        return eval_result

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction step to collect individual loss components and binary predictions during evaluation"""
        if not hasattr(self, '_eval_lm_losses'):
            self._eval_lm_losses = []
        if not hasattr(self, '_eval_binary_losses'):
            self._eval_binary_losses = []
        if not hasattr(self, '_eval_binary_preds'):
            self._eval_binary_preds = []
        if not hasattr(self, '_eval_binary_targets'):
            self._eval_binary_targets = []

        has_binary_images = 'images_binary' in inputs
        binary_images = inputs.pop('images_binary', None) if has_binary_images else None

        with torch.no_grad():
            outputs = model(**inputs)

        loss = None
        logits = None
        labels = inputs.get("labels", None)

        if "loss" in outputs:
            loss = outputs["loss"]
        elif hasattr(outputs, 'loss'):
            loss = outputs.loss

        if "logits" in outputs:
            logits = outputs["logits"]
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if hasattr(outputs, 'lm_loss') and hasattr(outputs, 'binary_loss'):
            lm_loss = outputs.lm_loss.detach().item() if hasattr(outputs.lm_loss, 'item') else float(outputs.lm_loss)
            binary_loss = outputs.binary_loss.detach().item() if hasattr(outputs.binary_loss, 'item') else float(outputs.binary_loss)

            self._eval_lm_losses.append(lm_loss)
            self._eval_binary_losses.append(binary_loss)

        if 'binary_labels' in inputs and hasattr(outputs, 'binary_loss'):
            try:

                binary_targets = inputs['binary_labels']
                if binary_targets.dtype == torch.bfloat16:
                    binary_targets = binary_targets.float()
                binary_targets = binary_targets.detach().cpu().numpy()

                if hasattr(outputs, 'binary_logits') and outputs.binary_logits is not None:
                    binary_logits = outputs.binary_logits

                    if binary_logits.dtype == torch.bfloat16:
                        binary_logits = binary_logits.float()

                    binary_probs = torch.sigmoid(binary_logits).detach().cpu().numpy()

                    self._eval_binary_preds.append(binary_probs.flatten())
                    self._eval_binary_targets.append(binary_targets.flatten())

            except Exception as e:
                logger.warning(f"[DEBUG] Failed to collect binary predictions: {e}")

        if prediction_loss_only:
            return loss, None, None

        return loss, logits, labels

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to log only losses (metrics removed)"""
        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        return eval_result

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """Override to log only losses (binary metrics removed)"""
        logs = super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
        return logs

    def log(self, logs: Dict[str, float]) -> None:
        """Override log method to add individual loss components to console output"""
        if hasattr(self, '_current_lm_loss'):
            logs['lm_loss'] = self._current_lm_loss
        if hasattr(self, '_current_binary_loss'):
            logs['binary_loss'] = self._current_binary_loss
        if hasattr(self, '_current_total_loss'):
            logs['total_loss'] = self._current_total_loss

        super().log(logs)

    def _prepare_inputs(self, inputs):
        """Override to handle binary labels for dual-loss training"""
        inputs = super()._prepare_inputs(inputs)

        if 'binary_labels' in inputs:
            inputs['binary_labels'] = inputs['binary_labels'].to(self.args.device)

        return inputs

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
