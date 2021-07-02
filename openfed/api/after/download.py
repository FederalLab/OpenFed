from openfed.common.logging import logger

from ..step import Backend, Step, after_download


class AfterDownload(Step):
    step_name = after_download

    def step(self, backend: Backend, flag: bool) -> None:
        if flag:  # Download success
            # download is to check others upload version
            if backend.reign.upload_version <= backend.version:
                logger.warning(
                    f"Excepted @{backend.version}, received @{backend.reign.upload_version}, discard.")
                return
            # Fetch data from federated core
            packages = backend.reign.tensor_indexed_packages
            task_info = backend.reign.task_info

            # Add them to aggregator
            for aggregator in backend.aggregator:
                aggregator.step(packages, task_info)

            # Increase the total number of received models
            backend.received_numbers += 1
            # Record current reign_task_info, which can be used in the following steps.
            backend.reign_task_info = task_info

            logger.info(f"{backend.received_numbers} at v.{backend.version} from {backend.nick_name}.")
        else:
            logger.debug(
                f"Try to download {backend.received_numbers+1} failed.")
