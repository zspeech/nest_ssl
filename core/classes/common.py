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
Common utilities and classes to replace nemo.core.classes.common
"""

from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict
from abc import ABC
import functools


@dataclass
class PretrainedModelInfo:
    """Information about a pretrained model."""
    pretrained_model_name: str
    description: str
    location: str
    class_definition: Optional[Any] = None


class Typing(ABC):
    """
    An interface which endows module with neural types.
    Simplified version.
    """
    
    @property
    def input_types(self) -> Optional[Dict[str, Any]]:
        """Define these to enable input neural type checks"""
        return None
    
    @property
    def output_types(self) -> Optional[Dict[str, Any]]:
        """Define these to enable output neural type checks"""
        return None


def typecheck(input_types=None, output_types=None):
    """
    Decorator for type checking. Simplified version - just returns the function.
    In a full implementation, this would perform actual type checking.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

