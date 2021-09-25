# @Author            : FederalLab
# @Date              : 2021-09-25 16:53:17
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:53:17
# Copyright (c) FederalLab. All rights reserved.
from collections import defaultdict
from typing import Any, Dict, List


def meta_reduce(meta_list: List[Dict[str, Any]],
                reduce_keys: List[str]) -> Dict:
    total_instances = sum([meta['instances'] for meta in meta_list])
    weight = [meta['instances'] / total_instances for meta in meta_list]

    reduce_meta = defaultdict(lambda: 0.0)

    for w, meta in zip(weight, meta_list):
        for k, v in meta.items():
            if k in reduce_keys:
                reduce_meta[k] += v * w

    return reduce_meta
