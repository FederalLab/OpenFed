from openfed.common.logging import logger

from ..step import Backend, Step, at_first


class AtFirst(Step):
    step_name = at_first

    def step(self, backend: Backend, *args, **kwargs) -> None:
        ...
