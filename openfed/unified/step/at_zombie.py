from openfed.common.logging import logger
from openfed.unified.step.base import Backend, Step


class AtZombie(Step):
    step_name = 'at_zombie'

    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        logger.debug(f"Waiting response from {backend.reign}")
