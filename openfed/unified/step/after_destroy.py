from abc import abstractmethod

from openfed.unified.step.base import Backend, Step


class AfterDestroy(Step):
    step_name = 'after_destroy'

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        ...
