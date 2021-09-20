# Copyright (c) FederalLab. All rights reserved.
from typing import List, Optional

from openfed.federated import collaborator, is_aggregator, is_collaborator
from openfed.utils import openfed_class_fmt


class FederatedOptimizer(object):

    def __init__(self,
                 optimizer,
                 role: str = collaborator,
                 max_acg_step: int = -1):
        self.optimizer = optimizer
        self.role = role
        self.max_acg_step = max_acg_step

    @property
    def aggregator(self) -> bool:
        return is_aggregator(self.role)

    @property
    def collaborator(self) -> bool:
        return is_collaborator(self.role)

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
        if self.collaborator:
            self._collaborator_step(*args, **kwargs)
        else:
            self._aggregator_step(*args, **kwargs)
        return self.optimizer.step()

    def _collaborator_step(self, *args, **kwargs):
        ...

    def _aggregator_step(self, *args, **kwargs):
        ...

    def round(self):
        if self.collaborator:
            return self._collaborator_round()
        else:
            return self._aggregator_round()

    def _collaborator_round(self):
        ...

    def _aggregator_round(self):
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
        return openfed_class_fmt.format(
            class_name=self.__class__.__name__, description=description)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)
