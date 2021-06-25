from abc import abstractmethod

from openfed.unified.step.base import Backend, Step


class AfterDownload(Step):
    step_name = 'after_download'

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> None:
        ...
