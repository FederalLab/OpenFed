import time
from typing import Generator, TypeVar

import openfed

from ..common import logger
from ..utils import openfed_class_fmt
from .deliver import Delivery
from .federated_world import FederatedWorld, ProcessGroup, Store
from .inform import Informer
from .register import register
from .world import World

_R = TypeVar("_R", bound='Reign')


class Reign(Informer, Delivery):
    """Contains all communication functions in Reign.
    """
    store: Store
    pg: ProcessGroup
    world: World
    federated_world: FederatedWorld

    # the request version number
    version: int

    def __init__(self,
                 store: Store,
                 pg: ProcessGroup,
                 federated_world: FederatedWorld,
                 world: World,
                 ):
        self.pg = pg
        self.store = store
        self.federated_world = federated_world
        self.world = world

        Informer.__init__(self)
        Delivery.__init__(self)

        self.version = 0
        self.set("version", self.version)

    def upload(self) -> bool:
        """Upload packages date to the other end.
        """

        if self.world.queen:
            # 1. set version number
            self.set('version', self.version)
            # 2. set pushing
            self.pushing()
            # 3. waiting king to response
            tic = time.time()
            while not self.is_pulling:
                toc = time.time()
                if toc-tic > openfed.SLEEP_VERY_LONG_TIME:
                    if openfed.VERBOSE.is_verbose:
                        logger.error("Timeout while uploading data ot server.")
                    return False
                if self.is_offline:
                    if openfed.VERBOSE.is_verbose:
                        logger.error("The server has shutted down.")
                    return False
                time.sleep(openfed.SLEEP_SHORT_TIME)
            else:
                self.push()

            # 4. set self to ZOMBINE
            self.zombine()

            return True
        else:
            # 1. write verion
            self.set('version', self.version)
            # 2. set pushing
            self.pushing()
            # 3. response
            if not self.is_pulling:
                # server will not wait for client in any time.
                if openfed.DEBUG.is_debug:
                    logger.error("Wrong state of client")
                return False
            else:
                self.push()
            # 4. set zombine
            self.zombine()
            return True

    def download(self) -> bool:
        """Download packages from other end.
        """
        # 1. set version
        self.set('version', self.version)
        if self.world.queen:
            # 2. set pulling
            self.pulling()
            # 3. wait and download
            tic = time.time()
            while not self.is_pushing:
                toc = time.time()
                if toc-tic > openfed.SLEEP_VERY_LONG_TIME:
                    if openfed.VERBOSE.is_verbose:
                        logger.error("Downloading failed, time out.")
                    return False
                if self.is_offline:
                    if openfed.VERBOSE.is_verbose:
                        logger.error("The server has shutted down.")
                    return False
                time.sleep(openfed.SLEEP_SHORT_TIME)
            else:
                self.pull()
        else:
            # 2. set pulling
            self.pulling()
            # 3. download data
            if not self.is_pushing:
                # server will not wait for client in any time.
                if openfed.DEBUG.is_debug:
                    logger.error("Wrong state of client")
                return False
            else:
                self.pull()
        # 4. set zombine
        self.zombine()
        return True

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Reign",
            description=(
                f"Version: {self.version}\n"
                f"Status: {self._get_state()}\n"
            )
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
