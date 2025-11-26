# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Simplified serialization utilities to replace nemo.core.classes.Serialization
"""

from typing import Dict, Any
from omegaconf import DictConfig


class Serialization:
    """
    Utility class for serialization/deserialization.
    Simplified version.
    """
    
    @staticmethod
    def from_config_dict(config: Dict[str, Any]):
        """
        Instantiate an object from a configuration dictionary.
        Supports Hydra-style instantiation with _target_ key.
        """
        if config is None:
            return None
        
        if isinstance(config, DictConfig):
            config = dict(config)
        
        if '_target_' in config:
            # Hydra-style instantiation
            target = config['_target_']

            # Remap NeMo targets to local implementations so we don't import nemo.*
            target_map = {
                # Preprocessor and encoder
                'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor': 'modules.audio_preprocessing.AudioToMelSpectrogramPreprocessor',
                'nemo.collections.asr.modules.ConformerEncoder': 'modules.conformer_encoder.ConformerEncoder',
                # SSL-specific modules
                'nemo.collections.asr.modules.MultiSoftmaxDecoder': 'modules.ssl_modules.multi_softmax_decoder.MultiSoftmaxDecoder',
                'nemo.collections.asr.modules.RandomBlockMasking': 'modules.ssl_modules.masking.RandomBlockMasking',
                'nemo.collections.asr.modules.RandomProjectionVectorQuantizer': 'modules.ssl_modules.quantizers.RandomProjectionVectorQuantizer',
                # Loss
                'nemo.collections.asr.losses.MultiMLMLoss': 'losses.ssl_losses.mlm.MultiMLMLoss',
            }

            # Apply remapping if needed
            target = target_map.get(target, target)

            parts = target.split('.')
            class_name = parts[-1]
            module_path = '.'.join(parts[:-1])

            import importlib
            try:
                module = importlib.import_module(module_path)
                class_obj = getattr(module, class_name)
            except (ImportError, AttributeError):
                raise ValueError(f"Could not import {target}")

            # Remove _target_ from config
            config = {k: v for k, v in config.items() if k != '_target_'}
            return class_obj(**config)
        else:
            # Direct instantiation (not supported in simplified version)
            return None

