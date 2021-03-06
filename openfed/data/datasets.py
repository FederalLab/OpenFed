# @Author            : FederalLab
# @Date              : 2021-09-25 16:51:52
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:51:52
# Copyright (c) FederalLab. All rights reserved.

from typing import Any, List

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from .partitioner import Partitioner


class FederatedDataset(Dataset):
    """Federated Dataset is the common used dataset in OpenFed.

    Each federated dataset has an unique part id. This is useful while doing
    simulation experiments.
    """
    part_id: int = 0
    total_parts: int = 1

    def set_part_id(self, part_id):
        assert 0 <= part_id < self.total_parts, \
            f'part_id({part_id}) out of range [0, {self.total_parts}])'
        self.part_id = part_id

    @property
    def total_samples(self) -> int:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__ + (
            f'(total_parts: {self.total_parts}, '
            f'total_samples: {self.total_samples}, '
            f'current_parts: {self.part_id})')


class PartitionerDataset(FederatedDataset):
    """PartitionerDataset can make a non-federated dataset as a federated
    dataset via specify different partition methods. It is useful while
    exploring the influence of non-iid experiments.

    Args:
        dataset: Any torch kind dataset.
        total_parts: How many parts the dataset participated into.
        partitioner: The way to participate the dataset.
    """
    dataset: Any
    parts_index_list: List
    partitioner: Partitioner

    def __init__(self, dataset: Any, total_parts: int,
                 partitioner: Partitioner):
        self.dataset = dataset
        self.total_parts = total_parts
        self.partitioner = partitioner

        self.parts_index_list = self.partitioner(total_parts,
                                                 self.data_index_list())

    def data_index_list(self) -> List:
        """Rewrite for your dataset.

        If dataset.classes is not existed, you should rewrite this method.
        """
        if not hasattr(self.dataset, 'classes') and not hasattr(
                self.dataset, 'targets'):
            raise RuntimeError(
                'Dataset does not contain `classes` and `targets`.'
                'Please rewrite this method for your dataset.')
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

    @property
    def total_samples(self):
        return len(self.dataset)
