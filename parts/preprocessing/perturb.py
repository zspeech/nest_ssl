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
Simplified audio perturbation utilities to replace nemo.collections.asr.parts.preprocessing.perturb
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod


class AudioAugmentor(ABC):
    """Base class for audio augmentation."""
    
    @abstractmethod
    def perturb(self, data):
        """Apply augmentation to data."""
        pass


class NoOpAugmentor(AudioAugmentor):
    """No-op augmentor that does nothing."""
    
    def perturb(self, data):
        """No-op: return data unchanged."""
        return data


class WhiteNoisePerturbation(AudioAugmentor):
    """Add white noise to audio."""
    
    def __init__(self, min_level: int = -90, max_level: int = -46):
        """
        Initialize white noise perturbation.
        
        Args:
            min_level: Minimum noise level in dB
            max_level: Maximum noise level in dB
        """
        self.min_level = min_level
        self.max_level = max_level
    
    def perturb(self, audio_segment):
        """Add white noise to audio segment."""
        noise_level_db = np.random.uniform(self.min_level, self.max_level)
        noise_level_linear = 10 ** (noise_level_db / 20.0)
        
        # Generate white noise
        noise = torch.randn_like(audio_segment.samples) * noise_level_linear
        
        # Add noise to audio
        audio_segment.samples = audio_segment.samples + noise
        
        return audio_segment


def process_augmentations(
    augmentor_cfg: Optional[Dict[str, Any]],
    global_rank: int = 0,
    world_size: int = 1,
) -> Optional[AudioAugmentor]:
    """
    Process augmentation configuration and return AudioAugmentor instance.
    
    Simplified version - returns None if no augmentations specified.
    """
    if augmentor_cfg is None:
        return None
    
    # For now, return None as we don't need complex augmentations for SSL training
    # This can be extended later if needed
    return None

