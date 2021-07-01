from openfed.common.logging import logger

from ..base import Backend, Step, before_destroy


class BeforeDestroy(Step):
    step_name = before_destroy

    def step(self, backend: Backend, *args, **kwargs) -> bool:
        # logger.debug(f"Try to destroy {backend.reign}.")
        return True
