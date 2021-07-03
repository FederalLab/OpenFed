# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os

import h5py
import numpy as np
import torch
from openfed.common import logger

from ..datasets import FederatedDataset
from ..utils import *

DEFAULT_CLIENTS_NUM = 3400
DEFAULT_BATCH_SIZE = 20
DEFAULT_TRAIN_FILE = 'fed_emnist_train.h5'
DEFAULT_TEST_FILE = 'fed_emnist_test.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_IMAGE = 'pixels'
_LABEL = 'label'


class EMNIST(FederatedDataset):
    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None, download: bool = True):
        data_file = os.path.join(
            root, DEFAULT_TRAIN_FILE if train else DEFAULT_TEST_FILE)

        if not os.path.isfile(data_file):
            if download:
                url = 'https://fedml.s3-us-west-1.amazonaws.com/fed_emnist.tar.bz2'
                logger.debug(f"Download dataset from {url} to {root}")
                if wget_https(url, root):
                    if tar_xvf(os.path.join(root, "fed_emnist.tar.bz2"), output_dir=root):
                        logger.debug("Downloaded.")
                else:
                    raise RuntimeError("Download dataset failed.")
            else:
                raise FileNotFoundError(f"{data_file} not exists.")

        data_h5 = h5py.File(data_file, "r")

        client_ids = list(data_h5[_EXAMPLE].keys())

        self.total_parts = len(client_ids)

        part_data_list = []
        part_target_list = []
        unique_label_set = set()
        for client_id in client_ids:
            part_data_list.append(
                np.array(data_h5[_EXAMPLE][client_id][_IMAGE][()]))
            part_target_list.append(
                np.array(data_h5[_EXAMPLE][client_id][_LABEL][()]).squeeze())
            unique_label_set.update(part_target_list[-1].tolist())
        self.part_data_list = part_data_list
        self.part_target_list = part_target_list

        self.transform = transform
        self.target_transform = target_transform

        self.classes = unique_label_set

    def __len__(self) -> int:
        return len(self.part_data_list[self.part_id])

    def __getitem__(self, index: int):
        data, target = self.part_data_list[self.part_id][index], self.part_target_list[self.part_id][index]

        target = torch.tensor(target).long()

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def total_samples(self):
        return sum([len(x) for x in self.part_data_list])
