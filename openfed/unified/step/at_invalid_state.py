from loguru import logger
from openfed.unified.step.base import Backend, Step


class AtInvalidState(Step):
    step_name = 'at_invalid_state'

    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        logger.error("An invalid state was encountered.")