from openfed.common.logging import logger

from ..step import Backend, Step, after_destroy


class AfterDestroy(Step):
    step_name = after_destroy

    def step(self, backend: Backend, *args, **kwargs) -> None:
        ...
