# Copyright (c) FederalLab. All rights reserved.
from typing import List, Tuple

import numpy as np
from openfed.utils import tablist


def digest(federated_dataset, verbose: bool = True) -> List:
    total_parts = federated_dataset.total_parts
    parts_list = []
    for p in range(total_parts):
        federated_dataset.set_part_id(p)
        parts_list.append(len(federated_dataset))
    if verbose:
        rdict = dict(
            Parts=total_parts,
            Samples=sum(parts_list),
            Mean=np.mean(parts_list),
            Var=np.var(parts_list),
        )
        print(
            tablist(
                head=list(rdict.keys()),
                data=list(rdict.values()),
                force_in_one_row=True,
            ))
    return parts_list


def label_distribution(federated_dataset,
                       top_k: int = 10) -> Tuple[List, List]:
    assert hasattr(federated_dataset, 'classes') and isinstance(
        federated_dataset.classes, int
    ), "Only classification federated dataset with specified the class number is supported."
    parts_list = digest(federated_dataset, verbose=False)

    top_k = min(len(parts_list), top_k)
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
