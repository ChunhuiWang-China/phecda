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

import numpy as np
import torch.utils.data

class PhecdaDataset(torch.utils.data.Dataset):
    """Dataset that provides for batching."""

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def collater(self, samples):
        """
        Get a list of samples to form to collate
        :param samples(List[dict]): sample to collate
        :return: dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError


# class PhecdaIterableDataset(torch.utils.data.IterableDataset):
#     ""
