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


from abc import abstractmethod
from typing import Any, List
from torch import Tensor
import numpy as np
from torch.utils.data import Dataset

from .partitioner import Partitioner


class FederatedDataset(Dataset):
    part_id: int = 0
    total_parts: int = 1

    def set_part_id(self, part_id):
        assert 0 <= part_id <= self.total_parts
        self.part_id = part_id

    @abstractmethod
    def total_samples(self):
        raise NotImplementedError


class PartitionerDataset(FederatedDataset):
    dataset: Dataset
    parts_index_list: List[np.array]
    partitioner: Partitioner

    def __init__(self,
                 dataset: Dataset,
                 total_parts: int,
                 partitioner: Partitioner):
        """
        Args:
            dataset: 
        """
        self.dataset = dataset
        self.total_parts = total_parts
        self.partitioner = partitioner

        self.parts_index_list = self.partitioner(
            total_parts, self.data_index_list())

    def data_index_list(self) -> List[np.array]:
        """Rewrite for your dataset. If dataset.classes is not existed, you should rewrite this method for your dataset.
        """
        if not hasattr(self.dataset, "classes") and not hasattr(self.dataset, "targets"):
            raise RuntimeError("Dataset does not contain `classes` and `targets`."
                               "Please rewrite this method for your dataset.")
        if isinstance(self.dataset.classes, int):
            classes = range(self.dataset.classes)
        else:
            classes = range(len(self.dataset.classes))
        if isinstance(self.dataset.targets, Tensor):
            targets = self.dataset.targets.numpy()
        else:
            targets = self.dataset.targets

        return [np.where(targets == cls)[0] for cls in classes]

    def __len__(self):
        return len(self.parts_index_list[self.part_id])

    def __getitem__(self, index: int) -> Any:
        idx = self.parts_index_list[self.part_id][index]
        return self.dataset[idx]

    def total_samples(self):
        return len(self.dataset)
