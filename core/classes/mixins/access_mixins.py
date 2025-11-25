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
Simplified access mixin to replace nemo.core.classes.mixins.access_mixins
"""

from typing import Dict, Any, Optional
from omegaconf import DictConfig


class AccessMixin:
    """
    Mixin for accessing intermediate layer outputs.
    Simplified version - minimal functionality.
    """
    
    def __init__(self):
        self._access_enabled = False
        self._access_cfg = {}
        self._module_registry = {}
        self._model_guid = None
    
    def is_access_enabled(self, guid: Optional[str] = None) -> bool:
        """Check if access is enabled."""
        return self._access_enabled
    
    def set_access_enabled(self, access_enabled: bool, guid: Optional[str] = None):
        """Enable or disable access."""
        self._access_enabled = access_enabled
    
    def get_module_registry(self, module) -> Dict[str, Any]:
        """Get module registry (simplified - returns empty dict)."""
        return {}
    
    def reset_registry(self):
        """Reset the registry."""
        self._module_registry = {}


def set_access_cfg(cfg: DictConfig, model_guid: str):
    """Set access configuration."""
    pass

