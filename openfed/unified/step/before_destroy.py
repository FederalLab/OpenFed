from openfed.unified.step.base import Backend, Step
from loguru import logger

class BeforeDestroy(Step):
    step_name = 'before_destroy'

    def __call__(self, backend: Backend, *args, **kwargs) -> bool:
        logger.debug(f"Try to destroy {backend.reign}.")
        return True