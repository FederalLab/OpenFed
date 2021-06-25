from abc import abstractmethod

from openfed.unified.step.base import Backend, Step


class AtNewEpisode(Step):
    step_name = 'at_new_episode'

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        ...
