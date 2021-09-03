# Copyright (c) FederalLab. All rights reserved.
import random
import time
from typing import Dict, List, Union

from openfed.core import DefaultMaintainer

from .const import (after_destroy, after_download, after_upload, at_failed,
                    at_first, at_invalid_state, at_last, at_new_episode,
                    at_zombie, before_destroy, before_download, before_upload)


def count_step(counts: Union[List[int], int]):
    _default_maintainer = DefaultMaintainer._default_maintainer

    assert _default_maintainer, 'Define a maintainer and use `with maintainer` context.'

    if isinstance(counts, int):
        counts = [
            counts,
        ]

    if _default_maintainer.leader:

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
    _default_maintainer = DefaultMaintainer._default_maintainer

    assert _default_maintainer, 'Define a maintainer and use `with maintainer` context.'

    if _default_maintainer.leader:

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


def dispatch_step(counts: Union[List[int], int], parts_list: Dict[str, int]):
    _default_maintainer = DefaultMaintainer._default_maintainer

    assert _default_maintainer, 'Define a maintainer and use `with maintainer` context.'

    if isinstance(counts, int):
        counts = [
            counts,
        ]
    parts_list_key = list(parts_list.keys())
    parts_list_value = list(parts_list.values())

    if _default_maintainer.leader:
        idx = 0
        pending_queue = random.sample(
            parts_list_value[idx],  # type: ignore
            counts[idx])  # type: ignore

        def before_upload_hook(maintainer) -> bool:
            nonlocal pending_queue

            if len(pending_queue) > 0:
                # assign a new task
                part_id = pending_queue.pop(-1)
                maintainer.meta['version'] = maintainer.version
                maintainer.meta['part_id'] = part_id
                maintainer.meta['mode'] = parts_list_key[idx]
                return True

            if len(maintainer.meta_list) < parts_list_value[idx]:
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
