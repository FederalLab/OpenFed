from abc import abstractmethod

from openfed.unified.step.base import Backend, Step


class AtLast(Step):
    step_name = 'at_last'

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        ...
