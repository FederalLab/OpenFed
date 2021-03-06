# @Author            : FederalLab
# @Date              : 2021-09-25 16:51:58
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:51:58
# Copyright (c) FederalLab. All rights reserved.
from abc import abstractmethod
from copy import deepcopy
from typing import List

import numpy as np
import numpy.random as np_random


class Partitioner(object):
    r"""Base class for partitioner.
    """

    @abstractmethod
    def partition(self, total_parts: int, data_index_list: List) -> List:
        raise NotImplementedError

    def __call__(self, total_parts: int, data_index_list: List) -> List:
        return self.partition(total_parts, data_index_list)


class PowerLawPartitioner(Partitioner):
    r"""PowerLawPartitioner

    Args:
        min_samples:
        min_classes:
        mean:
        sigma:

    Examples::

        >>> from torchvision.datasets import MNIST
        >>> from torchvision.transforms import ToTensor
        >>> from openfed.data import PartitionerDataset, samples_distribution
        >>> from openfed.data import PowerLawPartitioner
        >>> mnist = MNIST(r'/tmp/', True, ToTensor(), download=True)
        >>> dataset = PartitionerDataset(
        ...     mnist, total_parts=100, partitioner=PowerLawPartitioner())
        >>> samples_distribution(dataset)
        +-------+---------+--------+-----------+
        | Parts | Samples |  Mean  |    Var    |
        +-------+---------+--------+-----------+
        |  100  |  33814  | 338.14 | 136232.34 |
        +-------+---------+--------+-----------+
    """

    def __init__(self,
                 min_samples: int = 10,
                 min_classes: int = 2,
                 mean: float = 0.0,
                 sigma: float = 2.0):
        """mean and sigma is used for lognormal."""
        self.min_samples = min_samples
        self.min_classes = min_classes
        self.mean = mean
        self.sigma = sigma

    def partition(self, total_parts: int, data_index_list: List) -> List:
        """Refer to: https://github.com/litian96/FedProx/"""
        min_samples = self.min_samples
        min_classes = self.min_classes

        samples = max(min_samples // min_classes, 1)

        classes = len(data_index_list)
        parts_index_list = [[] for _ in range(total_parts)]
        cursor = [0 for _ in range(classes)]
        for p in range(total_parts):
            for c in range(min_classes):
                label = (p + c) % classes
                data_index = data_index_list[label]
                start, end = cursor[label], cursor[label] + samples
                parts_index_list[p] += data_index[start:end].tolist()
                cursor[label] = end

        # power law
        props = np_random.lognormal(
            self.mean,
            self.sigma,
            size=(classes, total_parts // classes, min_classes))
        normalized_props = props / np.sum(props, (1, 2), keepdims=True)

        for p in range(total_parts):
            for c in range(min_classes):
                label = (p + c) % classes
                data_index = data_index_list[label]
                data_samples = len(data_index)
                num_samples = (data_samples - cursor[label]
                               ) * normalized_props[label, p % classes, c]
                num_samples = max(1, int(num_samples))
                if cursor[label] + num_samples <= data_samples:
                    start, end = cursor[label], cursor[label] + num_samples
                    parts_index_list[p] += data_index[start:end].tolist()
                    cursor[label] = end
        return [np.array(p) for p in parts_index_list]


class DirichletPartitioner(Partitioner):
    r'''Dirichlet partitioner.

    Args:
        alpha:
        min_samples:

    Examples::

        >>> from openfed.data import DirichletPartitioner, PartitionerDataset,\
        >>> samples_distribution
        >>> from torchvision.datasets import MNIST
        >>> from torchvision.transforms import ToTensor
        >>> dataset = PartitionerDataset(
        ...     MNIST(r'/tmp/', True, ToTensor(), download=True),
        ...         total_parts=10, partitioner=DirichletPartitioner())
        >>> samples_distribution(dataset, True)
        +-------+---------+---------+----------+
        | Parts | Samples |   Mean  |   Var    |
        +-------+---------+---------+----------+
        |   10  |  60000  | 6000.00 | 23904.20 |
        +-------+---------+---------+----------+
        [5891, 5955, 6220, 5986, 5965, 6174, 6190, 5873, 6047, 5699]
    '''
    _MAX_LOOP = 10000

    def __init__(self, alpha: float = 100, min_samples: int = 10):
        """Obtain sample index list for each client from the Dirichlet
        distribution.

            This LDA method is first proposed by:
            Measuring the Effects of Non-Identical Data Distribution for
            Federated Visual Classification
            (https://arxiv.org/pdf/1909.06335.pdf).

            This can generate nonIIDness with unbalance sample number
            in each label.
            The Dirichlet distribution is a density over a K dimensional
            vector p whose K components are positive and sum to 1.
            Dirichlet can support the probabilities of a K-way
            categorical event.
            In FL, we can view K clients' sample number obeys
            the Dirichlet distribution.
            For more details of the Dirichlet distribution,
            please check https://en.wikipedia.org/wiki/Dirichlet_distribution
        Args:
            alpha: a concentration parameter controlling
            the identicalness among clients.
        """

        self.alpha = alpha
        self.min_samples = min_samples

    def dirichlet_partition(self, total_samples: int, total_parts: int,
                            parts_index_list: List, data_index: List):
        data_index = deepcopy(data_index)
        np_random.shuffle(data_index)
        # using dirichlet distribution to determine the unbalanced proportion
        # for each partition (total_parts in total).
        # e.g., when total_parts=4, proportions=[0.29, 0.38, 0.32, 0.00],
        # sum(proportions) = 1
        proportions = np_random.dirichlet(np.repeat(self.alpha, total_parts))

        # get the index in data_index according to the dirichlet distribution
        proportions = np.array([
            p * (len(idx) < total_samples / total_parts)
            for p, idx in zip(proportions, parts_index_list)
        ])
        normalized_proportions = proportions / sum(proportions)
        proportions = (np.cumsum(normalized_proportions) *
                       len(data_index)).astype(int)[:-1]

        # generate new list for each partition
        parts_index_list = [
            idx_j + idx.tolist() for idx_j, idx in zip(
                parts_index_list, np.split(data_index, proportions))
        ]
        return parts_index_list

    def __call__(self, total_parts: int, data_index_list: List) -> List:
        min_samples = self.min_samples
        total_samples = sum([len(x) for x in data_index_list])
        minimum_samples = -1
        loop_cnt = 0
        while minimum_samples < min_samples:
            parts_index_list = [[] for _ in range(total_parts)]

            for data_index in data_index_list:
                parts_index_list = self.dirichlet_partition(
                    total_samples, total_parts, parts_index_list, data_index)
                minimum_samples = min([len(p) for p in parts_index_list])
            loop_cnt += 1
            if loop_cnt > self._MAX_LOOP:
                raise RuntimeError(
                    f'Exceed maximum loop times: {self._MAX_LOOP}.')

        return [np.array(p) for p in parts_index_list]  # type: ignore


class IIDPartitioner(Partitioner):
    r'''IID partitioner.

    Examples::
        >>> from openfed.data import IIDPartitioner, PartitionerDataset,\
        >>> samples_distribution
        >>> from torchvision.datasets import MNIST
        >>> from torchvision.transforms import ToTensor
        >>> dataset = PartitionerDataset(
            MNIST(r'/tmp/', True, ToTensor(), download=True), total_parts=10,
                partitioner=IIDPartitioner())
        >>> samples_distribution(dataset, True)
        +-------+---------+---------+------+
        | Parts | Samples |   Mean  | Var  |
        +-------+---------+---------+------+
        |   10  |  59960  | 5996.00 | 0.00 |
        +-------+---------+---------+------+
        [5996, 5996, 5996, 5996, 5996, 5996, 5996, 5996, 5996, 5996]
    '''

    def partition(self, total_parts: int, data_index_list: List) -> List:
        parts_index_list = [[] for _ in range(total_parts)]
        for p in range(total_parts):
            for data_index in data_index_list:
                data_samples = len(data_index)
                step = data_samples // total_parts
                step = max(1, step)
                start, end = p * step, (p + 1) * step
                parts_index_list[p] += data_index[start:end].tolist()

        return [np.array(p) for p in parts_index_list]
