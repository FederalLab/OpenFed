from abc import abstractmethod

from openfed.unified.step.base import Backend, Step


class AtZombie(Step):
    step_name = 'at_zombie'

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        ...
