# @Author            : FederalLab
# @Date              : 2021-09-25 16:55:12
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:55:12
# Copyright (c) FederalLab. All rights reserved.
from threading import Thread
from typing import Any, Callable, Dict, Optional

from tqdm import trange

from openfed.core import Maintainer
from openfed.optim import FederatedOptimizer


class API(Thread):
    r'''API: Provides an API to handle backend logistics.

    Args:
        maintainer: The maintainer.
        fed_optim: The federated optimizer.
        rounds: The rounds to loop.
        agg_func: The agg function.
        reduce_func: The reduce function.
        kwargs: thread arguments.

    Example::

        >>> api = openfed.API(mt, fed_optim, rounds, agg_func)
        >>> # 1. run it on thread
        >>> api.start()
        >>> api.join()
        >>> # 2. run it on main process.
        >>> api.run()
    '''

    def __init__(self,
                 maintainer: Maintainer,
                 fed_optim: FederatedOptimizer,
                 rounds: int,
                 agg_func: Callable,
                 agg_func_kwargs: Dict[str, Any] = dict(),
                 reduce_func: Optional[Callable] = None,
                 reduce_func_kwargs: Dict[str, Any] = dict(),
                 with_test_round: bool = False,
                 **kwargs):
        super(API, self).__init__(**kwargs)

        self.maintainer = maintainer
        self.fed_optim = fed_optim
        self.rounds = rounds
        self.agg_func = agg_func
        self.agg_func_kwargs = agg_func_kwargs
        self.reduce_func = reduce_func
        self.reduce_func_kwargs = reduce_func_kwargs
        self.with_test_round = with_test_round

    def run(self):
        """A aggregator logistics."""
        maintainer = self.maintainer
        rounds = self.rounds
        fed_optim = self.fed_optim
        agg_func = self.agg_func
        agg_func_kwargs = self.agg_func_kwargs
        reduce_func = self.reduce_func
        reduce_func_kwargs = self.reduce_func_kwargs
        if maintainer.aggregator:
            process = trange(rounds)
            for r in process:
                # Train phase
                maintainer.package(fed_optim)
                maintainer.step()
                fed_optim.zero_grad()
                agg_func(
                    data_list=maintainer.data_list,
                    meta_list=maintainer.meta_list,
                    optim_list=fed_optim,
                    **agg_func_kwargs)
                fed_optim.step()
                fed_optim.round()

                fed_optim.clear_state_dict()

                if reduce_func:
                    train_info = reduce_func(maintainer.meta_list,
                                             **reduce_func_kwargs)

                maintainer.clear()

                # Test phase
                if self.with_test_round:
                    maintainer.package(fed_optim)
                    maintainer.step()

                    if reduce_func:
                        test_info = reduce_func(maintainer.meta_list,
                                                **reduce_func_kwargs)

                    maintainer.clear()

                maintainer.update_version()
                if reduce_func:
                    info = f'train: {train_info}' + \
                        f' test: {test_info}' if self.with_test_round else ''
                    process.set_description(info)
