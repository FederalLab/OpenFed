from openfed.common.logging import logger

from ..base import Backend, Step


class AtFailed(Step):
    step_name = 'at_failed'

    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        logger.debug("Failed at previous step.")
