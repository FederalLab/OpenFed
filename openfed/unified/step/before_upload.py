from abc import abstractmethod

from openfed.unified.step.base import Backend, Step


class BeforeUpload(Step):
    step_name = 'before_upload'

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> bool:
        ...
