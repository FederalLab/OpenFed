import time
from datetime import timedelta
from typing import Callable, Generator, List, Tuple, TypeVar

import openfed
from openfed.common.exception import ConnectTimeout, DeviceOffline
from openfed.common.logging import logger
from openfed.common.vars import ASYNC_OP
from openfed.federated.country import Country, ProcessGroup, Store
from openfed.federated.deliver import Delivery
from openfed.federated.inform import Informer
from openfed.federated.register import register
from openfed.federated.world import World
from openfed.utils import openfed_class_fmt
from torch._C._distributed_c10d import Work

_R = TypeVar("_R", bound='Reign')


class Reign(Informer, Delivery):
    """Contains all communication functions in Reign.
    """
    store: Store
    pg: ProcessGroup
    world: World
    country: Country

    # the request version number
    version: int

    # handler, step function, timestamp
    _download_hang_up: Tuple[Work, Callable, int]
    _upload_hang_up: Tuple[Work, Callable, int]

    def __init__(self,
                 store: Store,
                 pg: ProcessGroup,
                 country: Country,
                 world: World,
                 ):
        self.pg = pg
        self.store = store
        self.country = country
        self.world = world

        Informer.__init__(self)
        Delivery.__init__(self)

        self.version = 0
        self.set("version", self.version)

        self._download_hang_up = []
        self._upload_hang_up = []

    @property
    def upload_hang_up(self) -> bool:
        return len(self._upload_hang_up) > 0

    @property
    def download_hang_up(self) -> bool:
        return len(self._download_hang_up) > 0

    def deal_with_hang_up(self) -> bool:
        """Dealing with the handler for hang up operations.
        """
        if self.upload_hang_up:
            handler, step_func, timestamp = self._upload_hang_up
            if not self.is_offline:
                if self.world.is_queen:
                    # set self state
                    self.pushing()

                    # wait
                    tic = time.time()
                    while not self.is_pulling:
                        if self.is_offline:
                            raise DeviceOffline(str(self))
                        if timedelta(seconds=time.time() - tic) > openfed.DEFAULT_PG_TIMEOUT:
                            raise ConnectTimeout(str(self))
                        time.sleep((openfed.SLEEP_SHORT_TIME))
                    else:
                        # transfer
                        handler.wait()
                else:
                    # check state first
                    if self.is_pulling:
                        # set state
                        self.pushing()

                        # transfer
                        handler.wait()

        elif self.download_hang_up:
            handler, step_func, timestamp = self._download_hang_up
            if not self.is_offline:
                if self.world.is_queen:
                    # set self state
                    self.pulling()

                    # wait
                    tic = time.time()
                    while not self.is_pushing:
                        if self.is_offline:
                            raise DeviceOffline(str(self))
                        if timedelta(seconds=time.time() - tic) > openfed.DEFAULT_PG_TIMEOUT:
                            raise ConnectTimeout(str(self))
                        time.sleep((openfed.SLEEP_SHORT_TIME))
                    else:
                        # transfer
                        handler.wait()
                else:
                    # check state first
                    if self.is_pushing:
                        # set state
                        self.pulling()

                        # transfer
                        handler.wait()
        else:
            raise RuntimeError("No handler!")

        if self.is_offline:
            raise DeviceOffline(str(self))

        if handler.is_completed():
            step_func()
            self.zombine()
            if self.upload_hang_up:
                self._upload_hang_up = []
            else:
                self._download_hang_up = []
            return True
        else:
            if timedelta(seconds=time.time() - timestamp) > openfed.DEFAULT_PG_TIMEOUT:
                raise ConnectTimeout(f"Timeout while waiting {self}")
            else:
                # keep waiting
                return False

    def upload(self) -> bool:
        """Upload packages date to the other end.
        """
        # 1. set version number
        self.set('version', self.version)

        # 3. transfer
        if ASYNC_OP.is_async_op:
            handle, step_func = self.push()
            # store the necessary message, and hang up begining time.
            self._upload_hang_up = [handle, step_func, time.time()]
            return False
        else:
            if self.world.is_queen:
                # 2. set pulling
                self.pushing()
                # 3. check state
                tic = time.time()
                while not self.is_pulling:
                    if self.is_offline:
                        raise DeviceOffline(str(self))
                    if timedelta(seconds=time.time() - tic) > openfed.DEFAULT_PG_TIMEOUT:
                        raise ConnectTimeout(str(self))
                    time.sleep(openfed.SLEEP_SHORT_TIME)
            else:
                # 2. check state
                if not self.is_pulling:
                    if openfed.DEBUG.is_debug:
                        raise RuntimeError("Client state is not correct.")
                    else:
                        logger.error("Client state is not correct.")
                    return False
                # 3. set state
                self.pushing()
            # transfer
            self.push()
            self.zombine()
            return True

    def download(self) -> bool:
        """Download packages from other end.
        """
        # 1. set version
        self.set('version', self.version)

        # 3. transfer
        if ASYNC_OP.is_async_op:
            handle, step_func = self.pull()
            self._download_hang_up = [handle, step_func, time.time()]
            return False
        else:
            if self.world.is_queen:
                # 2. set pulling
                self.pulling()
                # 3. check state
                tic = time.time()
                while not self.is_pushing:
                    if self.is_offline:
                        raise DeviceOffline(str(self))
                    if timedelta(seconds=time.time() - tic) > openfed.DEFAULT_PG_TIMEOUT:
                        raise ConnectTimeout(str(self))
                    time.sleep(openfed.SLEEP_SHORT_TIME)
            else:
                # 2. check state
                if not self.is_pushing:
                    if openfed.DEBUG.is_debug:
                        raise RuntimeError("Client state is not correct.")
                    else:
                        logger.error("Client state is not correct.")
                    return False
                # 3. set state
                self.pulling()
            # transfer
            self.pull()
            self.zombine()
            return True

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Reign",
            description=f"Version: {self.version}"
            f"Status: {self._get_state().value}"
        )

    @classmethod
    def reign_generator(cls) -> Generator[_R, None, None]:
        """Return a generator to iterate over all reigns.
        """
        for _, world in register:
            if world is None and not world.ALIVE:
                continue
            for pg, reign in world:
                if reign is None and not world.ALIVE:
                    continue
                yield reign
                world._current_pg = pg

    @classmethod
    def default_reign(cls) -> _R:
        """Return the only reign. If more then one, raise warning.
        """
        if len(register) == 0:
            logger.warning("Empty world.")
        if len(register) > 1:
            if openfed.VERBOSE.is_verbose:
                logger.info(
                    "More than one register world, use the earliest one.")
        return register.default_world.default_reign
