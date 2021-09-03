# Copyright (c) FederalLab. All rights reserved.
from typing import List, Optional

from openfed.federated import follower, is_follower, is_leader
from openfed.utils import openfed_class_fmt


class FederatedOptimizer(object):
    def __init__(self,
                 optimizer,
                 role: str = follower,
                 max_acg_step: int = -1):
        self.optimizer = optimizer
        self.role = role
        self.max_acg_step = max_acg_step

    @property
    def leader(self) -> bool:
        return is_leader(self.role)

    @property
    def follower(self) -> bool:
        return is_follower(self.role)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def state_dict(self):
        return self.optimizer.state_dict()

    def acg(self, *args, **kwargs):
        ...

    def _acg_step(self, *args, **kwargs):
        ...

    def step(self, *args, **kwargs):
        if self.follower:
            self._follower_step(*args, **kwargs)
        else:
            self._leader_step(*args, **kwargs)
        return self.optimizer.step()

    def _follower_step(self, *args, **kwargs):
        ...

    def _leader_step(self, *args, **kwargs):
        ...

    def round(self):
        if self.follower:
            return self._follower_round()
        else:
            return self._leader_round()

    def _follower_round(self):
        ...

    def _leader_round(self):
        ...

    def clear_state_dict(
        self,
        keys: Optional[List] = None,
    ):
        if keys:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    for k in keys:
                        if k in state:
                            del state[k]

    def __repr__(self):
        description = repr(self.optimizer)
        return openfed_class_fmt.format(class_name=self.__class__.__name__,
                                        description=description)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)
