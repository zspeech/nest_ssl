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
Manifest processing utilities - extended version
"""

import json
import os
from typing import Iterator, List, Union, Optional, Dict, Any, Callable
from pathlib import Path


def item_iter(
    manifests_files: Union[str, List[str]],
    parse_func: Optional[Callable[[str, Optional[str]], Dict[str, Any]]] = None
) -> Iterator[Dict[str, Any]]:
    """
    Iterate over manifest items.
    
    Args:
        manifests_files: Path(s) to manifest file(s)
        parse_func: Optional function to parse each line
    
    Yields:
        Dictionary items from manifest
    """
    if isinstance(manifests_files, str):
        manifests_files = [manifests_files]
    
    for manifest_file in manifests_files:
        manifest_file = Path(manifest_file).expanduser().resolve()
        
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
        
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if parse_func:
                    item = parse_func(line, str(manifest_file))
                else:
                    item = __parse_item(line, str(manifest_file))
                
                if item:
                    yield item


def __parse_item(line: str, manifest_file: str) -> Dict[str, Any]:
    """Parse a single manifest line."""
    item = json.loads(line)
    
    # Audio file
    if 'audio_filename' in item:
        item['audio_file'] = item.pop('audio_filename')
    elif 'audio_filepath' in item:
        item['audio_file'] = item.pop('audio_filepath')
    else:
        raise KeyError(f"No 'audio_filename' or 'audio_filepath' in manifest item: {item}")
    
    # Resolve audio file path
    audio_file = item['audio_file']
    if not os.path.isabs(audio_file):
        data_dir = os.path.dirname(manifest_file)
        audio_file_path = os.path.join(data_dir, audio_file)
        if os.path.isfile(audio_file_path):
            audio_file = os.path.abspath(audio_file_path)
        else:
            audio_file = os.path.abspath(os.path.expanduser(audio_file))
    item['audio_file'] = audio_file
    
    # Duration
    if 'duration' not in item:
        item['duration'] = None
    
    # Text
    if 'text' not in item:
        item['text'] = ""
    
    # Other fields
    item = dict(
        id=item.get('id', len(item)),
        audio_file=item['audio_file'],
        duration=item['duration'],
        text=item['text'],
        offset=item.get('offset', None),
        speaker=item.get('speaker', None),
        orig_sr=item.get('orig_sample_rate', None),
        token_labels=item.get('token_labels', None),
        lang=item.get('lang', None),
    )
    
    return item
