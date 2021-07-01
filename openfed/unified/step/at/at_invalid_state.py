from openfed.common.logging import logger

from ..base import Backend, Step


class AtInvalidState(Step):
    step_name = 'at_invalid_state'

    def step(self, backend: Backend, *args, **kwargs) -> None:
        logger.debug("An invalid state was encountered.")
