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

import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import open_clip


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.use_openclip = vision_tower.startswith("hf-hub:")

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            if self.use_openclip:
                self.cfg_only = type('Config', (), {
                    'hidden_size': 1536,
                    'image_size': 320,
                    'patch_size': 32,
                })()
            else:
                self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        if self.use_openclip:
            model_name = self.vision_tower_name.replace("hf-hub:", "")

            if "CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup" in model_name:
                model_arch = "convnext_large_d_320"
                pretrained_name = "laion2b_s29b_b131k_ft_soup"
            elif "convnext_large_d_320" in model_name:
                model_arch = "convnext_large_d_320"
                if "ft-soup" in model_name:
                    pretrained_name = "laion2b_s29b_b131k_ft_soup"
                else:
                    pretrained_name = "laion2b_s29b_b131k_ft"
            else:
                model_arch = model_name.split("/")[-1].split(".")[0]
                pretrained_name = model_name

            print(f"Loading OpenCLIP model: {model_arch} with pretrained: {pretrained_name}")
            self.vision_tower, _, self.image_processor = open_clip.create_model_and_transforms(
                model_arch,
                pretrained=pretrained_name,
                device="cpu"
            )
            self.openclip_preprocess = self.image_processor
        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

        if self.use_openclip and hasattr(self, 'vision_tower'):
            self._expected_dtype = torch.float32

    def to(self, *args, **kwargs):
        """Override to() to ensure vision tower dtype consistency"""
        result = super().to(*args, **kwargs)

        if self.use_openclip and hasattr(self, 'vision_tower') and self.vision_tower is not None:
            target_dtype = None
            if args:
                for arg in args:
                    if isinstance(arg, torch.dtype):
                        target_dtype = arg
                        break
            if target_dtype is None and 'dtype' in kwargs:
                target_dtype = kwargs['dtype']

            if target_dtype is not None:
                self.vision_tower = self.vision_tower.to(dtype=target_dtype)
                self._expected_dtype = target_dtype

        return result

    def feature_select(self, image_forward_outs):
        if self.use_openclip:
            return image_forward_outs
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]
            if self.select_feature == 'patch':
                image_features = image_features[:, 1:]
            elif self.select_feature == 'cls_patch':
                image_features = image_features
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
            return image_features

    def _extract_convnext_stage3_features(self, images):
        """Extract features from stage 3 of ConvNeXT CLIP model"""
        visual = self.vision_tower.visual

        if hasattr(visual.trunk.stem, '0') and hasattr(visual.trunk.stem[0], 'weight'):
            model_weight = visual.trunk.stem[0].weight
            model_weight_dtype = model_weight.dtype
            model_weight_device = model_weight.device
            model_bias_dtype = visual.trunk.stem[0].bias.dtype if visual.trunk.stem[0].bias is not None else "None"
            model_bias_device = visual.trunk.stem[0].bias.device if visual.trunk.stem[0].bias is not None else "None"
            input_dtype = images.dtype
            input_device = images.device

            dtype_mismatch = model_weight_dtype != input_dtype
            device_mismatch = model_weight_device != input_device
            bias_dtype_mismatch = (visual.trunk.stem[0].bias is not None and model_bias_dtype != input_dtype)
            bias_device_mismatch = (visual.trunk.stem[0].bias is not None and model_bias_device != input_device)

            if dtype_mismatch or device_mismatch or bias_dtype_mismatch or bias_device_mismatch:
                images = images.to(dtype=model_weight_dtype, device=model_weight_device)

        x = visual.trunk.stem(images)

        for i in range(3):
            x = visual.trunk.stages[i](x)

        x = visual.trunk.stages[3](x)

        batch_size = x.shape[0]
        x = x.view(batch_size, 1536, -1)
        x = x.permute(0, 2, 1)

        return x

    @torch.no_grad()
    def forward(self, images):
        if self.use_openclip:
            if hasattr(self, 'vision_tower') and self.vision_tower is not None:
                try:
                    model_dtype = next(self.vision_tower.parameters()).dtype
                except StopIteration:
                    model_dtype = self.dtype
            else:
                model_dtype = self.dtype

            if type(images) is list:
                image_features = []
                for image in images:
                    image_feature = self._extract_convnext_stage3_features(
                        image.to(device=self.device, dtype=model_dtype).unsqueeze(0)
                    )
                    image_features.append(image_feature.to(image.dtype))
            else:
                image_features = self._extract_convnext_stage3_features(
                    images.to(device=self.device, dtype=model_dtype)
                )
                image_features = image_features.to(images.dtype)
        else:
            if type(images) is list:
                image_features = []
                for image in images:
                    image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if not self.is_loaded:
            return torch.float16

        if self.use_openclip:
            try:
                return next(self.vision_tower.parameters()).dtype
            except StopIteration:
                return torch.float16
        else:
            return self.vision_tower.dtype

    @property
    def device(self):
        if not self.is_loaded:
            return torch.device('cpu')

        if self.use_openclip:
            try:
                return next(self.vision_tower.parameters()).device
            except StopIteration:
                return torch.device('cpu')
        else:
            return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        if self.use_openclip:
            return 1536
        else:
            return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        if self.use_openclip:
            return 10
        else:
            return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        if self.use_openclip:
            return 100
        else:
            return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
