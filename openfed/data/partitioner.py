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
from copy import deepcopy
from typing import List

import numpy as np


class Partitioner(object):

    @abstractmethod
    def partition(self, total_parts: int, data_index_list: List[np.array]) -> List[np.array]:
        raise NotImplementedError

    def __call__(self, total_parts: int, data_index_list: List[np.array]) -> List[np.array]:
        return self.partition(total_parts, data_index_list)


class PowerLawPartitioner(Partitioner):
    def __init__(self,
                 min_samples: int   = 10,
                 min_classes: int   = 2,
                 mean       : float = 0.0,
                 sigma      : float = 2.0):
        """
        mean and sigma is used for lognormal.
        """
        self.min_samples = min_samples
        self.min_classes = min_classes
        self.mean        = mean
        self.sigma       = sigma

    def partition(self, total_parts: int, data_index_list: List[np.array]) -> List[np.array]:
        """Refer to:
            https://github.com/litian96/FedProx/blob/master/data/mnist/generate_niid.py
        """
        min_samples = self.min_samples
        min_classes = self.min_classes

        samples = max(min_samples // min_classes, 1)

        classes          = len(data_index_list)
        parts_index_list = [[] for _ in range(total_parts)]
        cursor           = [0 for _ in range(classes)]
        for p in range(total_parts):
            for c in range(min_classes):
                l = (p + c) % classes
                data_index = data_index_list[l]
                start, end = cursor[l], cursor[l] + samples
                parts_index_list[p] += data_index[start:end].tolist()
                cursor[l] = end

        # power law
        props = np.random.lognormal(
            self.mean, self.sigma, size=(
                classes, total_parts//classes, min_classes)
        )
        normalized_props = props / np.sum(props, (1, 2), keepdims=True)

        for p in range(total_parts):
            for c in range(min_classes):
                l            = (p+c) % classes
                data_index   = data_index_list[l]
                data_samples = len(data_index)
                num_samples = (
                    data_samples - cursor[l]) * normalized_props[l, p % classes, c]
                num_samples = max(1, int(num_samples))
                if cursor[l] + num_samples <= data_samples:
                    start, end = cursor[l], cursor[l]+num_samples
                    parts_index_list[p] += data_index[start:end].tolist()
                    cursor[l] = end
        return [np.array(p) for p in parts_index_list]


class DirichletPartitioner(Partitioner):
    _MAX_LOOP = 1000

    def partition(self, alpha: float = 100, min_samples: int = 10):
        """
            Obtain sample index list for each client from the Dirichlet distribution.

            This LDA method is first proposed by :
            Measuring the Effects of Non-Identical Data Distribution for
            Federated Visual Classification (https://arxiv.org/pdf/1909.06335.pdf).

            This can generate nonIIDness with unbalance sample number in each label.
            The Dirichlet distribution is a density over a K dimensional vector p whose K components are positive and sum to 1.
            Dirichlet can support the probabilities of a K-way categorical event.
            In FL, we can view K clients' sample number obeys the Dirichlet distribution.
            For more details of the Dirichlet distribution, please check https://en.wikipedia.org/wiki/Dirichlet_distribution
        Args:
            alpha: a concentration parameter controlling the identicalness among clients.
        """

        self.alpha       = alpha
        self.min_samples = min_samples

    def dirichlet_partition(self, 
                total_samples   : int,
                total_parts     : int,
                parts_index_list: List[np.array],
                data_index      : List[np.array]): 
        data_index = deepcopy(data_index)
        np.random.shuffle(data_index)
        # using dirichlet distribution to determine the unbalanced proportion
        # for each partition (total_parts in total).
        # e.g., when total_parts=4, proportions=[0.29, 0.38, 0.32, 0.00],
        # sum(proportions) = 1
        proportions = np.random.dirichlet(np.repeat(self.alpha, total_parts))

        # get the index in data_index according to the dirichlet distribution
        proportions = np.array([p * (len(idx) < total_samples / total_parts)
                                for p, idx in zip(proportions, parts_index_list)])
        normalized_proportions = proportions / sum(proportions)
        proportions = (np.cumsum(normalized_proportions)
                       * len(data_index)).astype(int)[:-1]

        # generate new list for each partition
        parts_index_list = [idx_j + idx.tolist() for idx_j, idx in zip(
            parts_index_list, np.split(data_index, proportions))]
        return parts_index_list

    def __call__(self, 
        total_parts    : int,
        data_index_list: List[np.array]) -> List[np.array]: 
        min_samples     = self.min_samples
        total_samples   = sum([len(x) for x in data_index_list])
        minimum_samples = -1
        loop_cnt        = 0
        while minimum_samples < min_samples:
            parts_index_list = [[] for _ in range(total_parts)]

            for data_index in data_index_list:
                parts_index_list = self.dirichlet_partition(
                    total_samples, total_parts, parts_index_list, data_index
                )
                minimum_samples = min([len(p) for p in parts_index_list])
            loop_cnt += 1
            if loop_cnt > self._MAX_LOOP:
                raise RuntimeError(
                    f"Exceed maximum loop times: {self._MAX_LOOP}.")

        return [np.array(p) for p in parts_index_list]


class IIDPartitioner(Partitioner):
    def partition(self, total_parts: int, data_index_list: List[np.array]) -> List[np.array]:

        parts_index_list = [[] for _ in range(total_parts)]
        for p in range(total_parts):
            for data_index in data_index_list:
                data_samples = len(data_index)
                step         = data_samples     // total_parts
                step         = max(1, step)
                start, end = p * step, (p+1) * step
                parts_index_list[p] += data_index[start:end].tolist()

        return [np.array(p) for p in parts_index_list]
