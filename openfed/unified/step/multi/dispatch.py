from typing_extensions import overload
from typing import Dict, List

import numpy as np
from openfed.common import logger

from ..base import Backend, MultiStep


class Dispatch(MultiStep):
    """
    NOTE: If you choose this step function, `BeforeUpload` must be disabled.
    """
    pending_queue: List[int]

    # part_id -> nick name
    running_queue: Dict[int, str]
    finished_queue: Dict[int, str]

    @overload
    def __init__(self, total_parts: int, samples: int):
        """
        Args:
            total_parts: the total number of parts splitted in a simulation federated work.
            samples: the number of parts activated during simulation.
        """

    @overload
    def __init__(self, total_parts: int, sample_ratio: float):
        """
        Args:
            total_parts: the total number of parts splitted in a simulation federated work.
            sample_ratio: the ratio to be activated during simulation.
        """

    def __init__(self, total_parts: int, samples: int = None,  sample_ratio: float = None):
        assert not (
            samples is None and sample_ratio is None), "one of samples or sample_ratio must be specified."

        self.total_parts = total_parts
        self.samples = samples if samples else int(total_parts * sample_ratio)

        # Initialize queue
        self.reset()

        # Count the finished parts
        # If finished all parts in this round, reset inner part buffer.
        self._after_download()
        # Dispatch a part to be finished.
        self._before_upload()

    def _permutation(self):
        return np.random.permutation(self.total_parts)[
            :self.samples]

    def reset(self):
        self.pending_queue = self._permutation()
        self.running_queue = dict()
        self.finished_queue = dict()

    def after_download(self, backend: Backend, *args, **kwargs):
        task_info = backend.reign_task_info
        part_id = task_info["part_id"]

        logger.debug(f"Download a model from {backend.nick_name}.")

        # pop from running queue
        self.running_queue.pop(part_id)

        # add to finished queue
        self.finished_queue[part_id] = backend.nick_name

        # All finished
        if len(self.running_queue) == 0 and len(self.pending_queue) == 0:
            # Reset
            self.reset()
            logger.debug(f"Finished a round.")
        else:
            logger.debug(
                f"Pending: {len(self.pending_queue)}, Runing: {len(self.running_queue)}, Finished: {len(self.finished_queue)}")

    def before_upload(self, backend: Backend, *args, **kwargs) -> bool:
        # version is not used in dispatch mode
        if len(self.pending_queue) > 0:
            # assign a new part id
            part_id = self.pending_queue.pop(-1)

            # generate task_info
            task_info = dict(
                part_id=part_id,
                version=backend.version,
            )

            # set task_info
            backend.reign.set_task_info(task_info)

            # reset old state

            assert backend.optimizer
            assert backend.aggregator
            assert backend.state_dict

            # reset old state
            backend.reign.reset()

            # pack new data
            backend.reign.reset_state_dict(backend.state_dict)
            for aggregator, optimizer in zip(backend.aggregator, backend.optimizer):
                backend.reign.pack_state(aggregator)
                backend.reign.pack_state(optimizer)

            return True

        elif len(self.running_queue) > 0:
            # waiting other client to be finished.
            logger.debug(
                f"Waiting following client to submit there task: {list(self.running_queue.values())}.")
            return False
        else:
            # unknown case.
            return False
