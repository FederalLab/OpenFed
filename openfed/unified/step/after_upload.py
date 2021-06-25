from abc import abstractmethod

from openfed.unified.step.base import Backend, Step


class AfterUpload(Step):
    step_name = 'after_upload'

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        ...
