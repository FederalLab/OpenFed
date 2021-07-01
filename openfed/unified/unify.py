import time
from typing import Any, Callable, Dict, List

import openfed
from openfed.common import SafeTread, TaskInfo, logger
from openfed.federated import Destroy, Maintainer, Peeper, Reign, World
from openfed.utils import keyboard_interrupt_handle, openfed_class_fmt
from torch import Tensor

from .utils import after_connection, backend_access, before_connection


class Unify(Peeper):
    """Provide a unified api for backend and frontend.
    """
    maintainer: Maintainer = None

    reign: Reign = None

    world: World = None

    version: int = 0

    frontend: bool = True
    # fontend xor backward == True
    backend: bool = False

    async_op: bool = False

    dynamic_address_loading: bool = True

    state_dict: List[Dict[str, Tensor]] = None

    _hooks_for_informers: List[Callable] = None
    _hooks_for_delivers: List[Callable] = None

    def __init__(self,
                 frontend: bool = True,
                 capture_keyboard_interrupt: bool = True,
                 async_op_beckend: bool = True,
                 dynamic_address_loading: bool = True,
                 register_default_step_for_backend: bool = True):
        """Whether act as a frontend.
        Frontend is always in sync mode, which will ease the coding burden.
        Backend will be set as async mode by default.
        """
        self.frontend = frontend
        # Set a flag for backend.
        self.backend = not self.frontend
        if capture_keyboard_interrupt:
            logger.debug("OpenFed will capture keyboard interrupt signal.")
            keyboard_interrupt_handle()

        if self.frontend:
            self.async_op = False
        elif async_op_beckend:
            self.async_op = True
        else:
            self.async_op = False

        self.dynamic_address_loading = dynamic_address_loading
        self.register_default_step_for_backend = register_default_step_for_backend

        # Set default value
        self.version = 0

        self._hooks_for_delivers = []
        self._hooks_for_informers = []

    @before_connection
    def add_informer_hook(self, hook: Callable):
        self._hooks_for_informers.append(hook)

    @before_connection
    def add_deliver_hook(self, hook: Callable):
        self._hooks_for_delivers.append(hook)

    @after_connection
    def _add_hook_to_reign(self):
        for hook in self._hooks_for_informers:
            if hook.bounding_name not in self.reign._hook_dict:
                # register a clone of informer hook.
                # informer hook may contain some inner variable, which is not allowed
                # to share with each other.
                self.reign.register_collector(hook.clone())
        for hook in self._hooks_for_delivers:
            if hook not in self.reign._hook_list:
                # register the hook directly.
                # deliver hook is not allowed to have inner parameters.
                # it can be used among all reign.
                self.reign.register_cypher(hook)

    @property
    @after_connection
    def nick_name(self) -> str:
        return self.reign.nick_name

    def build_connection(self, *args, **kwargs):
        raise NotImplementedError

    def set_state_dict(self, state_dict: Dict[str, Tensor]):
        raise NotImplementedError

    def __repr__(self):
        return openfed_class_fmt.format(
            class_name="Unify",
            description="The unify api for backend and frontend."
        )

    @after_connection
    def finish(self):
        Destroy.destroy_all_in_a_world(self.world)
        if self.maintainer:
            self.maintainer.manual_stop()

        if not self.frontend:
            logger.info(f"Finished.\n {self}")
            exit(0)

    @backend_access
    @after_connection
    def run(self, *args, **kwargs):
        return SafeTread.run(self, *args, **kwargs)

    def __enter__(self):
        self.old_variable_list = [
            openfed.DYNAMIC_ADDRESS_LOADING.is_dynamic_address_loading,
            openfed.ASYNC_OP.is_async_op,
        ]

        openfed.DYNAMIC_ADDRESS_LOADING.set(self.dynamic_address_loading)
        openfed.ASYNC_OP.set(self.async_op)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore state
        dynamic_address_loading, async_op = self.old_variable_list

        openfed.DYNAMIC_ADDRESS_LOADING.set(dynamic_address_loading)
        openfed.ASYNC_OP.set(async_op)

    @after_connection
    def set_task_info(self, task_info: TaskInfo) -> None:
        self.reign.set_task_info(task_info)

    @after_connection
    def get_task_info(self) -> TaskInfo:
        return self.reign.task_info

    @after_connection
    def scatter(self):
        return self.reign.scatter()

    @after_connection
    def collect(self):
        return self.reign.collect()

    @after_connection
    def update_version(self, version: int = None):
        """Update inner model version.
        """
        if version:
            self.version = version
        else:
            self.version += 1

    @after_connection
    def set(self, key: str, value: Any) -> None:
        self.reign.set(key, value)

    @after_connection
    def get(self, key: str) -> Any:
        return self.reign.get(key)

    @after_connection
    def upload(self) -> bool:
        """As for frontend, it is much easier for us to judge the new version.
        A download and upload is build a version updating.
        So increase version number here.
        """
        self.collect()
        self.scatter()
        if self.frontend:
            return self._wait_handler(self.reign.upload(self.version))
        else:
            return self.reign.upload(self.version)

    @after_connection
    def download(self) -> bool:
        self.collect()
        self.scatter()
        if self.frontend:
            return self._wait_handler(self.reign.download(self.version))
        else:
            return self.reign.download(self.version)

    @after_connection
    def _wait_handler(self, flag: bool):
        if flag:
            return True
        elif openfed.ASYNC_OP.is_async_op:
            while not self.reign.deal_with_hang_up():
                if self.reign.is_offline:
                    return False
                time.sleep(openfed.SLEEP_SHORT_TIME.seconds)
            return True
