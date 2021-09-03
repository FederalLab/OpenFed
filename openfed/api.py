# Copyright (c) FederalLab. All rights reserved.
from typing import Callable, Optional

from tqdm import trange

from openfed.core import Maintainer
from openfed.optim import FederatedOptimizer


def api(maintainer: Maintainer,
        fed_optim: FederatedOptimizer,
        rounds: int,
        agg_func: Callable,
        reduce_func: Optional[Callable] = None,
        **kwargs):
    r"""Provides an API to handle backend logistics.

    Args:
        maintainer: The maintainer.
        fed_optim: The federated optimizer.
        rounds: The rounds to loop.
        agg_func: The agg function.
        reduce_func: The reduce function.
        kwargs: Additional arguments for agg func.

    .. node::
        You can use following way to pass additional arguments for reduce_func:
        
        >>> def decorated_reduce_func(reduce_func, **kwargs):
        >>>     def _reduce_func(*args):
        >>>         return reduce_func(*args, **kwargs)
        >>>     return _reduce_func
        >>> api(mt, fed_optim, rounds, agg_func,
        >>> decorated_reduce_func(reduce_func, **kwargs))
    """
    if maintainer.leader:
        process = trange(rounds)
        for r in process:
            maintainer.package(fed_optim)
            maintainer.step()
            fed_optim.zero_grad()
            agg_func(data_list=maintainer.data_list,
                     meta_list=maintainer.meta_list,
                     optim_list=fed_optim,
                     **kwargs)
            fed_optim.step()
            fed_optim.round()

            fed_optim.clear_state_dict()

            if reduce_func:
                process.set_description(str(reduce_func(maintainer.meta_list)))

            maintainer.update_version()
            maintainer.clear()
