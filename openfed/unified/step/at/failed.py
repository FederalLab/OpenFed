from openfed.common.logging import logger

from ..base import Backend, Step


class AtFailed(Step):
    step_name = 'at_failed'

    def step(self, backend: Backend, *args, **kwargs) -> None:
        logger.debug("Failed at previous step.")
