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
Simplified audio to text dataset utilities to replace nemo.collections.asr.data.audio_to_text_dataset
"""

import copy
from typing import Any, Dict, Optional
from omegaconf import DictConfig, open_dict

from utils.logging import get_logger
from data import audio_to_text

logger = get_logger(__name__)


def inject_dataloader_value_from_model_config(model_cfg: dict, dataloader_cfg: DictConfig, key: str):
    """
    Extracts a value from model config and propagates it to dataloader config.
    
    Args:
        model_cfg: Model configuration dictionary
        dataloader_cfg: Dataloader configuration (DictConfig)
        key: Key to extract from model_cfg and inject into dataloader_cfg
    """
    if key not in model_cfg:
        logger.info(
            f"Model level config does not contain `{key}`, "
            f"please explicitly provide `{key}` to the dataloaders."
        )
        return
    
    if not isinstance(dataloader_cfg, DictConfig):
        dataloader_cfg = DictConfig(dataloader_cfg)
    
    # If key exists in the data loader config
    if key in dataloader_cfg:
        # Dataloader key is provided and is non-null
        if dataloader_cfg[key] is not None and model_cfg[key] != dataloader_cfg[key]:
            # Model level key doesn't match Dataloader level key
            logger.warning(
                f'`{key}` is explicitly provided to the data loader, and is different from '
                f'the `{key}` provided at the model level config.\n'
                f'If this is incorrect, please set the dataloader\'s `{key}` to None.'
            )
        else:
            # Dataloader key is None or values match - propagate from model level
            with open_dict(dataloader_cfg):
                dataloader_cfg[key] = model_cfg[key]
    else:
        # If key doesn't exist in dataloader_cfg, inject it explicitly
        with open_dict(dataloader_cfg):
            dataloader_cfg[key] = model_cfg[key]


def get_char_dataset(config: dict, augmentor: Optional[Any] = None) -> audio_to_text.AudioToCharDataset:
    """
    Instantiates a Character Encoding based AudioToCharDataset.
    
    Args:
        config: Config of the AudioToCharDataset.
        augmentor: Optional AudioAugmentor object for augmentations on audio data.
    
    Returns:
        An instance of AudioToCharDataset.
    """
    if 'labels' not in config:
        logger.warning("dataset does not have explicitly defined labels")
    
    dataset = audio_to_text.AudioToCharDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config.get('labels', None),
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', False),
        parser=config.get('parser', 'en'),
        return_sample_id=config.get('return_sample_id', False),
        channel_selector=config.get('channel_selector', None),
    )
    return dataset


def get_chain_dataset(datasets, ds_config, rank):
    """
    Create a chain dataset from multiple datasets.
    Simplified version.
    """
    from torch.utils.data import ChainDataset
    return ChainDataset(datasets)

