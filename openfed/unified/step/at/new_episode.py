from openfed.common.logging import logger

from ..base import Backend, Step, at_new_episode


class AtNewEpisode(Step):
    step_name = at_new_episode

    def step(self, backend: Backend, *args, **kwargs) -> None:
        logger.debug("Start a new episode.")
