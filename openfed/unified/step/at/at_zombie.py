from openfed.common.logging import logger

from ..base import Backend, Step


class AtZombie(Step):
    step_name = 'at_zombie'

    def step(self, backend: Backend, *args, **kwargs) -> None:
        logger.debug(f"Waiting response from {backend.reign}")
