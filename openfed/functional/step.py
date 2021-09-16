# Copyright (c) FederalLab. All rights reserved.
import random
import time
from typing import Dict, List, Union

from openfed.core.const import DefaultMaintainer

from .const import (after_destroy, after_download, after_upload, at_failed,
                    at_first, at_invalid_state, at_last, at_new_episode,
                    at_zombie, before_destroy, before_download, before_upload)


def count_step(counts: Union[List[int], int]):
    r"""Stops the loop according to the number of received models.
    
    Args:
        counts: If the number of received models is greater than counts, stop
            the loop immediately and turn to next count.
        
    .. Example::

        >>> count_step(15) # Stop when receive the fifteen-th models.
        >>> # Stop when receive the fifth models.
        >>> # Then continue to receive the next fifteen models.
        >>> count_step([5, 15]) 
    """
    _default_maintainer = DefaultMaintainer._default_maintainer

    assert _default_maintainer, 'Define a maintainer and use `with maintainer` context.'

    if isinstance(counts, int):
        counts = [
            counts,
        ]

    if _default_maintainer.aggregator:

        def before_upload_hook(maintainer) -> bool:
            request_version = maintainer.pipe.meta.get('version')

            if request_version > maintainer.version:
                return False
            else:
                maintainer.meta['version'] = maintainer.version
                return True

        _default_maintainer.register_step_hook(nice=50,
                                               step_hook=before_upload_hook,
                                               step_name=before_upload)

        idx = 0

        def at_last_hook(maintainer):
            nonlocal idx
            if len(maintainer.meta_list) == counts[idx]:  # type: ignore
                maintainer.manual_stop()
                idx = idx + 1
                idx = len(counts) % idx  # type: ignore

        _default_maintainer.register_step_hook(nice=50,
                                               step_hook=at_last_hook,
                                               step_name=at_last)

        _default_maintainer.register_step_hook(nice=50,
                                               step_hook=lambda x: True,
                                               step_name=before_destroy)
        _default_maintainer.register_step_hook(nice=50,
                                               step_hook=lambda x: True,
                                               step_name=before_download)


def period_step(period: float):
    r"""Stops the loop period.
    
    Args:
        period: The period second time to stop.
        
    .. Example::

        >>> period_step(15) # Stop the loop every 15 seconds.
    """
    _default_maintainer = DefaultMaintainer._default_maintainer

    assert _default_maintainer, 'Define a maintainer and use `with maintainer` context.'

    if _default_maintainer.aggregator:

        def before_upload_hook(maintainer) -> bool:
            request_version = maintainer.pipe.meta.get('version')

            if request_version > maintainer.version:
                return False
            else:
                maintainer.meta['version'] = maintainer.version
                return True

        _default_maintainer.register_step_hook(nice=50,
                                               step_hook=before_upload_hook,
                                               step_name=before_upload)

        tic = time.time()

        def at_last_hook(maintainer):
            nonlocal tic
            if time.time() - tic > period:  # type: ignore
                maintainer.manual_stop()
                tic = time.time()

        _default_maintainer.register_step_hook(nice=50,
                                               step_hook=at_last_hook,
                                               step_name=at_last)

        _default_maintainer.register_step_hook(nice=50,
                                               step_hook=lambda x: True,
                                               step_name=before_destroy)
        _default_maintainer.register_step_hook(nice=50,
                                               step_hook=lambda x: True,
                                               step_name=before_download)


def dispatch_step(counts: Union[List[int], int], parts_list: Dict[str, List]):
    r"""Dispatch a part id from part list and stop based the count.
    
    Args:
        counts: If the number of received models is greater than counts, stop
            the loop immediately and turn to next count.
        parts_list: The part list to dispatch.
        
    .. Example::

        >>> dispatch_step(15, dict(train=list(range(100))))
        >>> dispatch_step([15, 20], dict(train=list(range(100), test=list(range(50)))))
    """
    _default_maintainer = DefaultMaintainer._default_maintainer

    assert _default_maintainer, 'Define a maintainer and use `with maintainer` context.'

    if isinstance(counts, int):
        counts = [
            counts,
        ]
    parts_list_key = list(parts_list.keys())
    parts_list_value = list(parts_list.values())

    if _default_maintainer.aggregator:
        idx = 0
        pending_queue = random.sample(parts_list_value[idx], counts[idx])

        def before_upload_hook(maintainer) -> bool:
            nonlocal pending_queue

            if len(pending_queue) > 0:
                # assign a new task
                part_id = pending_queue.pop(-1)
                maintainer.meta['version'] = maintainer.version
                maintainer.meta['part_id'] = part_id
                maintainer.meta['mode'] = parts_list_key[idx]
                return True

            if len(maintainer.meta_list) < counts[idx]:  # type: ignore
                # wait an unfinished task
                return False

            return False

        _default_maintainer.register_step_hook(nice=50,
                                               step_hook=before_upload_hook,
                                               step_name=before_upload)

        idx = 0

        def at_last_hook(maintainer):
            nonlocal idx
            if len(maintainer.meta_list) == counts[idx]:  # type: ignore
                maintainer.manual_stop()
                idx = idx + 1
                idx = len(counts) % idx  # type: ignore

                nonlocal pending_queue
                pending_queue = random.sample(
                    parts_list_value[idx],  # type: ignore
                    counts[idx])  # type: ignore

        _default_maintainer.register_step_hook(nice=50,
                                               step_hook=at_last_hook,
                                               step_name=at_last)

        _default_maintainer.register_step_hook(nice=50,
                                               step_hook=lambda x: True,
                                               step_name=before_destroy)
        _default_maintainer.register_step_hook(nice=50,
                                               step_hook=lambda x: True,
                                               step_name=before_download)
