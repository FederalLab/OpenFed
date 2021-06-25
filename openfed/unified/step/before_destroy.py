from abc import abstractmethod

from openfed.unified.step.base import Backend, Step


class BeforeDestroy(Step):
    step_name = 'before_destroy'

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> bool:
        ...
