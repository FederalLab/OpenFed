from loguru import logger
from openfed.unified.step.base import Backend, Step


class AtFailed(Step):
    step_name = 'at_failed'

    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        logger.error("Failed at previous step.")
