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

from verl.single_controller.base.worker import DistGlobalInfo, DistRankInfo, Worker


class MegatronWorker(Worker):
    def __init__(self, cuda_visible_devices=None) -> None:
        super().__init__(cuda_visible_devices)

    def get_megatron_global_info(self):
        from megatron.core import parallel_state as mpu

        tp_size = mpu.get_tensor_model_parallel_world_size()
        dp_size = mpu.get_data_parallel_world_size()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        info = DistGlobalInfo(tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, cp_size=cp_size)
        return info

    def get_megatron_rank_info(self):
        from megatron.core import parallel_state as mpu

        tp_rank = mpu.get_tensor_model_parallel_rank()
        dp_rank = mpu.get_data_parallel_rank()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        cp_rank = mpu.get_context_parallel_rank()
        info = DistRankInfo(tp_rank=tp_rank, dp_rank=dp_rank, pp_rank=pp_rank, cp_rank=cp_rank)
        return info

    def _init_hf_config_and_tf_config(
        self,
        model_path,
        tokenizer_or_path,
        dtype,
        override_model_config,
        override_transformer_config,
        trust_remote_code=False,
    ):
        from transformers import AutoConfig

        from verl.models.mcore import hf_to_mcore_config
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.utils.model import update_model_config

        # Step 1: initialize the tokenizer
        self.local_path = copy_to_local(model_path)
        if tokenizer_or_path is None:
            self.tokenizer = hf_tokenizer(self.local_path, trust_remote_code=trust_remote_code)
            self.processor = hf_processor(self.local_path, trust_remote_code=trust_remote_code)
        elif isinstance(tokenizer_or_path, str):
            self.tokenizer = hf_tokenizer(copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code)
            self.processor = hf_processor(copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code)
        else:
            self.tokenizer = tokenizer_or_path
            self.processor = tokenizer_or_path

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        # Step 2: get the hf
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Step 3: override the hf config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config.get("model_config", {}))
        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(hf_config, "architectures", None)
        if self.rank == 0:
            print(f"Model config after override: {hf_config}")
        tf_config = hf_to_mcore_config(hf_config, dtype, **override_transformer_config)

        def add_optimization_config_to_tf_config(tf_config):
            # add optimization config to tf_config, e.g. checkpointing
            if self.config.model.get("enable_gradient_checkpointing", False):
                gradient_checkpointing_cfg = dict(self.config.model.get("gradient_checkpointing_kwargs", dict()))
                tf_config.recompute_method = gradient_checkpointing_cfg.get("activations_checkpoint_method", "full")
                tf_config.recompute_granularity = gradient_checkpointing_cfg.get(
                    "activations_checkpoint_granularity", "full"
                )
                tf_config.recompute_num_layers = gradient_checkpointing_cfg.get("activations_checkpoint_num_layers", -1)
            if megatron_config := self.config.get("megatron", {}):
                if extra := megatron_config.get("extra", {}):
                    for k, v in extra.items():
                        setattr(tf_config, k, v)

        add_optimization_config_to_tf_config(tf_config)

        print(f"TF config: {tf_config}")
        self.hf_config = hf_config
        self.tf_config = tf_config
