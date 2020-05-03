# coding=utf-8
# Copyright (C) 2020 PHECDA AUTHORS; Chunhui Wang 
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
# ==============================================================================

import torch
import torch.nn.functional as F
import numpy as np
import sys
from . import PhecdaDataset

class AudioDataset(PhecdaDataset):
    def __init__(
            self,
            sample_rate,
            max_sample_size=None,
            min_sample_size=None,
            shuffle = True,
            min_length=0,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.size = []
        self.max_sample_size = (max_sample_size if max_sample_size is not None else sys.maxsize)
        self.min_sample_size = (min_sample_size if min_sample_size is not None else self.max_sample_size)
        self.min_length = min_length
        self.shuffle = shuffle

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        return len(self.size)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]

    def collater(self, samples):
        samples = [
            s for s in samples if s['source'] is not None and len(s["source"]) > 0
        ]

        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]
        target_size = min(min(sizes), self.max_sample_size)

        if target_size < self.min_length:
            return {}

        if self.min_sample_size < target_size:
            target_size = np.random.randint(self.min_sample_size, target_size + 1)

        collated_sources = sources[0].new(len(sources), target_size)
        for i, (source, size) in enumerate(len(sources), target_size):
            diff = size - target_size
            assert diff >= 0
            if diff == 0:
                collated_sources[i] = source
            else:
                collated_sources[i] = self.crop_to_max_size()