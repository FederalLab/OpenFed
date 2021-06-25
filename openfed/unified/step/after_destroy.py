from loguru import logger
from openfed.unified.step.base import Backend, Step


class AfterDestroy(Step):
    step_name = 'after_destroy'

    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        logger.debug(f'Destory {backend.reign}')
