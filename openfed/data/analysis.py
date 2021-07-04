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


from typing import List, Tuple

import numpy as np
from openfed.utils import tablist

from .datasets import FederatedDataset


class Analysis(object):
    @classmethod
    def digest(cls, federated_dataset: FederatedDataset, verbose: bool = True) -> List:
        total_parts = federated_dataset.total_parts
        parts_list = []
        for p in range(total_parts):
            federated_dataset.set_part_id(p)
            parts_list.append(len(federated_dataset))
        if verbose:
            rdict = dict(
                Parts   = total_parts,
                Samples = sum(parts_list),
                Mean    = np.mean(parts_list),
                Var     = np.var(parts_list),
            )
            print(tablist(
                head             = list(rdict.keys()),
                data             = list(rdict.values()),
                force_in_one_row = True,
            ))
        return parts_list

    @classmethod
    def label_distribution(cls, federated_dataset: FederatedDataset, top_k: int = 10) -> Tuple[List, List]:
        assert hasattr(federated_dataset, 'classes') and isinstance(federated_dataset.classes,
                                                                    int), "Only classification federated dataset with specified the class number is supported."
        parts_list = cls.digest(federated_dataset, verbose=False)

        top_k    = min(len(parts_list), top_k)
        part_ids = np.argsort(parts_list)[:top_k]

        # go through dataset, time comsuming.
        label_distribution_list = []
        for part_id in part_ids:
            federated_dataset.set_part_id(part_id)

            label_distribution = np.zeros(federated_dataset.classes)
            for _, l in federated_dataset:
                label_distribution[l] += 1
            label_distribution = label_distribution / \
                np.sum(label_distribution)
            label_distribution_list.append(label_distribution)
        return part_ids, label_distribution_list
