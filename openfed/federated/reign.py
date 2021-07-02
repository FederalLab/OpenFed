import time
from datetime import timedelta
from typing import Callable, Generator, Tuple, TypeVar

import openfed
from openfed.common import ASYNC_OP, logger
from openfed.utils import openfed_class_fmt, tablist
from torch._C._distributed_c10d import Work

from .deliver import Delivery
from .inform import Informer
from .space import Country, ProcessGroup, Store, World
from .utils.exception import ConnectTimeout, DeviceOffline, WrongState
from .utils.register import register

_R = TypeVar("_R", bound='Reign')


class Reign(Informer, Delivery):
    """Contains all communication functions in Reign.
    """
    store: Store
    pg: ProcessGroup
    world: World
    country: Country

    # handler, step function, timestamp
    _download_hang_up: Tuple[Work, Callable, int]
    _upload_hang_up: Tuple[Work, Callable, int]

    def __init__(self,
                 store: Store,
                 pg: ProcessGroup,
                 country: Country,
                 world: World,
                 ) -> None:
        self.pg = pg
        self.store = store
        self.country = country
        self.world = world

        Informer.__init__(self)
        Delivery.__init__(self)

        self._download_hang_up = []
        self._upload_hang_up = []

    @property
    def upload_version(self) -> int:
        return self.get("upload_version")

    @property
    def download_version(self) -> int:
        return self.get("download_version")

    @property
    def upload_hang_up(self) -> bool:
        return len(self._upload_hang_up) > 0

    @property
    def download_hang_up(self) -> bool:
        return len(self._download_hang_up) > 0

    def transfer(self,
                 to: bool,
                 handler: Work = None,
                 tic: float = None,
                 step_func: Callable = None) -> bool:
        """
        Args:
            to: it true, transfer data to other side.
            handler: if not None, call handler.wait().
            tic: start counting time.
        """
        if self.is_offline:
            raise DeviceOffline(self)

        def _state():
            return self.is_pulling if to else self.is_pushing

        # logic judge
        if self.world.follower:
            # set state first
            if to:
                self.pushing()
            else:
                self.pulling()
            # wait until satisfied
            tic = time.time() if not tic else tic

            while not _state():
                if self.is_offline:
                    raise DeviceOffline(self)
                toc = time.time()
                if timedelta(seconds=toc-tic) > openfed.DEFAULT_PG_TIMEOUT:
                    raise ConnectTimeout(self)
                time.sleep(openfed.SLEEP_SHORT_TIME.seconds)
        else:
            # check state first
            if not _state():
                raise WrongState(self)
            # set state
            if to:
                self.pushing()
            else:
                self.pulling()
        # transfer
        if handler:
            handler.wait()
            step_func()
        else:
            if to:
                self.push()
            else:
                self.pull()
        self.zombie()
        return True

    def deal_with_hang_up(self) -> bool:
        """Dealing with the handler for hang up operations.
        """
        if self.upload_hang_up:
            handler, step_func, tic = self._upload_hang_up
            to = True
        elif self.download_hang_up:
            handler, step_func, tic = self._download_hang_up
            to = False
        else:
            return False

        try:
            flag = self.transfer(to=to, handler=handler,
                                 tic=tic, step_func=step_func)
        except WrongState as e:
            return False

        if handler.is_completed():
            if self.upload_hang_up:
                self._upload_hang_up = []
            else:
                self._download_hang_up = []
            return flag
        else:
            toc = time.time()
            if timedelta(seconds=toc-tic) > openfed.DEFAULT_PG_TIMEOUT:
                raise ConnectTimeout(f"Timeout while waiting {self}")
            else:
                # keep waiting
                return False

    def upload(self, version: int) -> bool:
        """Upload packages date to the other end.
        """

        # set version on task info
        self.set("upload_version", version)

        if ASYNC_OP.is_async_op:
            handle, step_func = self.push()
            # store the necessary message, and hang up begining time.
            self._upload_hang_up = [handle, step_func, time.time()]
            return False
        else:
            return self.transfer(to=True)

    def download(self, version: int) -> bool:
        """Download packages from other end.
        """

        # set version
        self.set("download_version", version)

        if ASYNC_OP.is_async_op:
            handle, step_func = self.pull()
            self._download_hang_up = [handle, step_func, time.time()]
            return False
        else:
            return self.transfer(to=False)

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Reign",
            description=tablist(
                head=["Nick Name", "Version", "Status"],
                data=[self.nick_name, self.version, self._get_state().value],
            )
        )

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Reign",
            description=self.nick_name,
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
            logger.debug("Empty world.")
        if len(register) > 1:
            logger.debug("More than one register world, use the earliest one.")
        return register.default_world.default_reign
