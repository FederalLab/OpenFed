# Copyright (c) FederalLab. All rights reserved.
from typing import List

import numpy as np

from openfed.utils import tablist


def samples_distribution(federated_dataset, verbose: bool = True) -> List:
    r"""Generates a simple statistic information about the given dataset.

    Args:
        federated_dataset: The given dataset.
        verbose: If ``True``, print a digest information about the dataset.

    Returns:
        List contains each part's samples.
    """
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
