from abc import abstractmethod

from openfed.unified.step.base import Backend, Step


class AtFailed(Step):
    step_name = 'at_failed'

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        ...
