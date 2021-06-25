from abc import abstractmethod
from typing import Union


class Backend(object):
    ...


class Step(object):
    step_name: str

    @abstractmethod
    def __call__(self, backend: Backend, *args, **kwargs) -> Union[None, bool]:
        ...
