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
Simplified NeuralModule base class to replace nemo.core.NeuralModule
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class NeuralModule(nn.Module, ABC):
    """
    Base class for all neural modules.
    This is a simplified version of NeMo's NeuralModule.
    """
    
    def __init__(self):
        super().__init__()
    
    @property
    def input_types(self) -> Optional[Dict[str, Any]]:
        """Returns definitions of module input ports."""
        return None
    
    @property
    def output_types(self) -> Optional[Dict[str, Any]]:
        """Returns definitions of module output ports."""
        return None
    
    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    @classmethod
    def from_config_dict(cls, config: Dict[str, Any]):
        """
        Instantiate a module from a configuration dictionary.
        Supports Hydra-style instantiation with _target_ key.
        """
        if isinstance(config, dict) and '_target_' in config:
            # Hydra-style instantiation
            target = config['_target_']
            # Extract class from target string (e.g., "module.path.ClassName")
            parts = target.split('.')
            class_name = parts[-1]
            module_path = '.'.join(parts[:-1])
            
            # For now, we'll use a simple approach - import and instantiate
            # This is a simplified version - in production you might want more robust handling
            import importlib
            try:
                module = importlib.import_module(module_path)
                class_obj = getattr(module, class_name)
            except (ImportError, AttributeError):
                # Fallback: try to find in common locations
                # This is a simplified fallback - you may need to adjust based on your structure
                raise ValueError(f"Could not import {target}")
            
            # Remove _target_ from config before passing to constructor
            config = {k: v for k, v in config.items() if k != '_target_'}
            return class_obj(**config)
        else:
            # Direct instantiation
            return cls(**config)

