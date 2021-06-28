from openfed.common.logging import logger

from .base import Backend, Step


class BeforeDestroy(Step):
    step_name = 'before_destroy'

    def __call__(self, backend: Backend, *args, **kwargs) -> bool:
        logger.debug(f"Try to destroy {backend.reign}.")
        return True
