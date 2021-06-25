from loguru import logger
from openfed.unified.step.base import Backend, Step


class AtNewEpisode(Step):
    step_name = 'at_new_episode'

    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        logger.debug("Start a new episode.")
