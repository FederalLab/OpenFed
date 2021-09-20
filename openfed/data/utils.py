# Copyright (c) FederalLab. All rights reserved.
import numpy as np
from typing import List

from openfed.utils import tablist


def samples_distribution(federated_dataset, verbose: bool = True) -> List:
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
