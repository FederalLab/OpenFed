from openfed.common.logging import logger

from ..base import Backend, Step, at_first


class AtFirst(Step):
    step_name = at_first

    def step(self, backend: Backend, *args, **kwargs) -> None:
        ...
