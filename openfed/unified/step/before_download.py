from abc import abstractmethod

from openfed.unified.step.base import Backend, Step


class BeforeDownload(Step):
    step_name = 'before_download'

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> bool:
        ...
