from abc import abstractmethod

from openfed.unified.step.base import Backend, Step


class AtInvalidState(Step):
    step_name = 'at_invalid_state'

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        ...
