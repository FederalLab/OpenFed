from openfed.common.logging import logger

from .base import Backend, Step


class AfterDestroy(Step):
    step_name = 'after_destroy'

    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        logger.debug(f'Destory {backend.reign}')
