from openfed.common.logging import logger

from .base import Backend, Step


class BeforeDownload(Step):
    step_name = 'before_download'

    def __call__(self, backend: Backend, *args, **kwargs) -> bool:
        logger.debug(f"Try to download a new model from  {backend.reign}.")
        return True
