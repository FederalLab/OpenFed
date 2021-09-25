# @Author            : FederalLab
# @Date              : 2021-09-25 16:51:38
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:51:38
# Copyright (c) FederalLab. All rights reserved.
import time
from collections import defaultdict
from copy import deepcopy
from queue import PriorityQueue
from typing import Any, Callable, Dict, List, Optional, Union

from torch import Tensor

from openfed.common.meta import Meta
from openfed.federated import (FederatedProperties, Pipe, init_federated_group,
                               is_aggregator, is_collaborator)
from openfed.functional.const import (after_destroy, after_download,
                                      after_upload, at_failed, at_first,
                                      at_invalid_state, at_last,
                                      at_new_episode, at_zombie,
                                      before_destroy, before_download,
                                      before_upload)
from openfed.utils import FMT, tablist
from .const import DefaultMaintainer
from .functional import fed_context


class Maintainer(object):
    r'''The user interface for OpenFed.

    Args:
        fed_props: The federated group belongs to.
        state_dict: Indicates tensors exchanged. If not specified, you should
            load it via :func:``load_state_dict``. Default: ``None``

    Example::

        >>> mt = Maintainer(fed_props, network.state_dict(keep_vars=True))

        >>> print(mt)
        <OpenFed> Maintainer
        +------------------+-----------+-------+
        |       role       | nick_name | pipes |
        +------------------+-----------+-------+
        | openfed_collaborator |   client  |   1   |
        +------------------+-----------+-------+
        >>> with mt:
        >>>     ...
    '''
    pipe: Pipe
    pipes: List[Pipe]
    current_step: str

    fed_props: FederatedProperties

    _package_hooks: Any
    _unpackage_hooks: Any
    _step_hooks: Dict[str, Any]

    def __init__(self,
                 fed_props: FederatedProperties,
                 state_dict: Optional[Any] = None):
        self.fed_props = fed_props

        # call while package
        self._package_hooks = PriorityQueue()
        # call while unpackage
        self._unpackage_hooks = PriorityQueue()
        self._step_hooks = defaultdict(PriorityQueue)

        self.version: int = 0
        self.stopped: bool = False
        self.received_numbers: int = 0
        self.last_aggregate_time: float = time.time()

        self.meta: Meta = Meta()
        self.meta_list: List[Meta] = []

        # The data just received
        self.data: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.data_list: List[Dict[Tensor, Any]] = []

        # The data need to be sent
        self.packaged_data: Dict[str, Dict[str, Any]] = defaultdict(dict)

        self.state_dict: Dict[str, Any] = state_dict or defaultdict(dict)

        self.pipes: List[Pipe] = []
        # set an initialize pipe
        self.pipe: Pipe = None  # type: ignore

        if self.fed_props:
            self.build_connection()

    @property
    def role(self) -> str:
        return self.fed_props.role

    @property
    def aggregator(self) -> bool:
        return is_aggregator(self.role)

    @property
    def collaborator(self) -> bool:
        return is_collaborator(self.role)

    @property
    def nick_name(self) -> str:
        return self.fed_props.nick_name

    @property
    def anti_nick_name(self) -> str:
        r'''Returns nick name of the other side.
        '''
        assert self.pipe
        return self.pipe.nick_name

    def register_package_hook(self, nice: int, package_hook: Callable):
        r'''Register a package hook.

        Args:
            nice: Priority of the hook, ``0`` is the first hook to be
                registered.
            package_hook: Package hook. It should take in `state` and `p`,
                returns modified `state`.

        .. note::
            All package hooks will be called in :func:``upload``.

        Example::

            >>> def package(state, p):
            >>>     for k, v in state.items():
            >>>         if v is not None:
            >>>             state[k] = v.to(p)
            >>>     return state
            >>> maintainer.register_package_hook(nice=10, package_hook=package)
        '''
        self._package_hooks.put((nice, package_hook))

    def register_unpackage_hook(self, nice: int, unpackage_hook: Callable):
        r'''Register an unpackage hook.

        Args:
            nice: Priority of the hook, ``0`` is the first hook to be
                registered.
            unpackage_hook: Package hook. It should take in `state` and `p`,
                returns modified `state`.

        .. note::
            All unpackage hooks will be called in :func:``download``.

        Example::

            >>> def unpackage(state, p):
            >>>     for k, v in state.items():
            >>>         if v is not None:
            >>>             state[k] = v.to(p)
            >>>     return state
            >>> maintainer.register_unpackage_hook(nice=10,
            >>>                                    package_hook=unpackage)
        '''
        self._unpackage_hooks.put((nice, unpackage_hook))

    def register_step_hook(self,
                           nice: int,
                           step_hook: Callable,
                           step_name: Optional[str] = None):
        r'''Register a step hook.

        Args:
            nice: Priority of the hook, ``0`` is the first hook to be
                registered.
            step_hook: Step hook. It should take in `maintainer`,
                returns ``None`` or ``bool``.
            step_name: When to apply this step hook.

        .. note::
            Step hook is used to control the behavior of `aggregator`.
            It will be called in :func:``_aggregator_step``.

        Example::

            >>> def before_upload_hook(maintainer) -> bool:
            >>>     request_version = maintainer.pipe.meta.get('version')

            >>>     if request_version > maintainer.version:
            >>>         return False
            >>>     else:
            >>>         maintainer.meta['version'] = maintainer.version
            >>>         return True

            >>> _default_maintainer.register_step_hook(nice=50,
            ...                                  step_hook=before_upload_hook,
            ...                                  step_name=before_upload)
        '''
        step_name = step_name or step_hook.step_name
        assert step_name, 'Step name must be a valid string.'

        self._step_hooks[step_name].put((nice, step_hook))

    def build_connection(self):
        r'''Builds connection to the given federated group.
        '''
        pipes = init_federated_group(self.fed_props)

        assert pipes, 'init_federated_group failed.'

        self.pipes += pipes

        self.pipe = pipes[0]

    def update_version(self, version: Optional[Any] = None):
        r'''Updates inner version.

        Args:
            version: The newer version. If not given, we will increment the
                inner version by 1. Default:``None``
        '''
        if version is not None:
            self.version = version
        else:
            self.version += 1
        self.meta['version'] = self.version

    def load_state_dict(self, state_dict: Dict[str, Tensor]):
        r'''Loads state dict to exchange with other end.

        Args:
            state_dict: State dict to exchange with.
        '''
        self.state_dict.update(state_dict)

    @fed_context
    def transfer(self, to: bool) -> Union[Any, None]:
        r'''Transfer data to another end.

        Args:
            to: If ``True``, upload the data to the other end. If ``False``,
                download data from the other end.

        Returns:
            If download was successful, return the downloaded data.
        '''
        if to:
            self.pipe.upload(self.packaged_data)
        else:
            return self.pipe.download()

    def download(self) -> bool:
        r'''Downloads data from the other end.
        '''
        # clear data before download
        self.data.clear()
        data = self.transfer(to=False)
        assert data

        for n, p in self.state_dict.items():
            p_data = data[n]

            # decode received data.
            for nice, hook in self._unpackage_hooks.queue:
                p_data = hook(p_data, p)

        self.data = data

        if self.aggregator:
            # convert to tensor index
            tensor_data = dict()
            for n, p in self.state_dict.items():
                tensor_data[p] = data[n]
            # cache the data
            self.data_list.append(tensor_data)
            self.meta_list.append(deepcopy(self.meta))

        return True

    def upload(self) -> bool:
        r'''Uploads data to the other end.
        '''
        assert self.packaged_data

        for n, p in self.state_dict.items():
            p_data = self.packaged_data[n]

            # apply various transformations, such as encryption here.
            for nice, hook in self._package_hooks.queue:
                p_data = hook(p_data, p)

        self.transfer(to=True)

        return True

    def step(self, *args, **kwargs):
        if self.collaborator:
            self._collaborator_step(*args, **kwargs)
        else:
            self._aggregator_step(*args, **kwargs)

    def _collaborator_step(self, *args, **kwargs):
        download = kwargs.pop('download', True)
        upload = kwargs.pop('upload', True)
        meta = kwargs.pop('meta', None)

        self.pipe.set_meta(meta or self.meta)

        if upload:
            self.upload()

        if download:
            self.download()

        self.meta = self.pipe.meta

        if meta:
            meta.update(self.meta)

    def _aggregator_step(self, *args, **kwargs):
        self.stopped = False

        def step(step_name: str, *args, **kwargs):
            self.current_step = step_name

            step_hook = self._step_hooks[step_name].queue

            output = []
            for nice, hook in step_hook:
                output.append(hook(self, *args, **kwargs))

            if False in output:
                return False
            elif True in output:
                return True
            else:
                return None

        while not self.stopped and len(self.pipes) > 0:
            step(at_new_episode)
            for i, pipe in enumerate(self.pipes):
                if self.stopped:
                    break

                self.pipe = pipe
                step(at_first)

                if pipe.is_offline:
                    step(before_destroy)
                    del self.pipes[i]
                    step(after_destroy, True)
                elif pipe.is_zombie:
                    step(at_zombie)
                elif pipe.is_pushing:
                    # collaborator pushes data to aggregator,
                    # as a aggregator, we need to download
                    if step(before_download):
                        flag = self.download()
                        step(after_download, flag)
                    else:
                        step(at_failed)
                elif pipe.is_pulling:
                    # collaborator pulls data to aggregator,
                    # as a aggregator, we need to upload
                    if step(before_upload):
                        flag = self.upload()
                        step(after_upload, flag)
                    else:
                        step(at_failed)
                else:
                    step(at_invalid_state)

                step(at_last)

            # sleep for a while to wait all the state have been correctly set.
            time.sleep(0.01)

    def package(self,
                optim_list: Optional[Any] = None,
                state_keys: Optional[List[str]] = None):
        r'''Package data to upload.

        Args:
            optim_list: The state of optim will be uploaded to.
            state_keys: The state keys to be uploaded. If ``None``, upload all
                state. Default: ``None``.
        '''
        self.packaged_data.clear()
        if optim_list and not isinstance(optim_list, list):
            optim_list = [
                optim_list,
            ]
        for n, p in self.state_dict.items():
            p_data = self.packaged_data[n]

            p_data['param'] = p
            if optim_list:
                for optim in optim_list:
                    if p in optim.state:
                        state = optim.state[p]
                        if state_keys is not None:
                            for key in state_keys:
                                if key in state:
                                    p_data[key] = state[key]
                        else:
                            for key in state.keys():
                                p_data[key] = state[key]

    def unpackage(self,
                  optim_list: Optional[Any] = None,
                  state_keys: Optional[List[str]] = None):
        """Unpackage state to optim.

        .. note::
            This operation will automatically load the `param` property to
            state_dict by a replace operation.

        Args:
            optim_list: The state of optim will be rewritten to.
            state_keys: The state keys to be rewritten. If ``None``,
                rewrite all state. Default: ``None``.
        """
        assert self.data
        if optim_list and not isinstance(optim_list, list):
            optim_list = [
                optim_list,
            ]
        for n, p in self.state_dict.items():
            p_data = self.data[n]

            p.copy_(p_data['param'])

            if optim_list:
                for optim in optim_list:
                    if p in optim.state:
                        state = optim.state[p]
                        if state_keys is not None:
                            for key in state_keys:
                                if key in state:
                                    state[key] = p_data[key]
                        else:
                            for key in state.keys():
                                if key in p_data:
                                    state[key] = p_data[key]

    def clear(self):
        r'''Clears inner cached data.
        '''
        self.data_list.clear()
        self.meta_list.clear()

    def __del__(self):
        self.manual_stop()
        self.pipes.clear()

    def manual_stop(self):
        r'''Stop while loop in :func:`_aggregator_step`.'''
        self.stopped = True

    def __enter__(self):
        self._default_maintainer = DefaultMaintainer._default_maintainer
        DefaultMaintainer._default_maintainer = self

    def __exit__(self, exc_type, exc_value, trace):
        DefaultMaintainer._default_maintainer = self._default_maintainer

    def __repr__(self):
        head = ['role', 'nick_name', 'pipes']
        data = [self.role, self.nick_name, len(self.pipes)]
        description = tablist(head, data, force_in_one_row=True)
        return FMT.openfed_class_fmt.format(
            class_name=self.__class__.__name__, description=description)
