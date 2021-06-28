import logging
import time
import warnings
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from openfed.common.constants import (DEFAULT_PG_LONG_TIMEOUT,
                                      DEFAULT_PG_SHORT_TIMEOUT,
                                      DEFAULT_PG_TIMEOUT)
from openfed.common.logging import logger
from openfed.utils import openfed_class_fmt
from torch._C._distributed_c10d import (BarrierOptions, PrefixStore,
                                        ProcessGroup, Store)
from torch.distributed.distributed_c10d import (Backend, P2POp,
                                                is_mpi_available,
                                                is_nccl_available)
from torch.distributed.rendezvous import rendezvous

from ..utils.exception import ConnectTimeout
from ..utils.lock import acquire_all, release_all

try:
    from torch.distributed.distributed_c10d import (ProcessGroupGloo,
                                                    ProcessGroupMPI,
                                                    ProcessGroupNCCL)
except ImportError:
    ...


class World():
    ...


class Country(object):
    """Warper all variables as a privacy namespace.
    """

    def __init__(self, world: World) -> None:
        """
        Args:
            world: the world this country belongs to.
        """
        self.world: World = world

        # Alias to self.WORLD for backward compatibility
        self.WORLD: Optional[ProcessGroup] = None
        self.NON_GROUP_MEMBER = object()

        # Cached process groups
        # For NCCL and GLOO pg, it is a map from ProcessGroup to (Backend, Store)
        # For MPI pg, it is a map from ProcessGroup to (Backend, None)
        self._pg_map: Dict[ProcessGroup, Tuple[str, Optional[Store]]] = {}
        # Keep trace of the point2point groups.
        self._point2point_groups: Dict[ProcessGroup,
                                       Tuple[str, Optional[Store]]] = {}
        # Process group's names, map from ProcessGroup to str
        self._pg_names: Dict[ProcessGroup, str] = {}
        # Process group's global rank to local rank mapping
        self._pg_group_ranks: Dict[ProcessGroup, Dict[int, int]] = {}

        # Default process group state
        self._default_pg_init_method = None

        # Process group count for default naming
        self._group_count = 0

        self.STORE_BASED_BARRIER_PREFIX = "store_based_barrier_key"

    def _store_based_barrier(self, rank: int, store: Store, timeout: timedelta):
        """
        Barrier based on store which is used for synchronizing processes after
        ``init_process_group`` or ``new_group``. Intended to be used only with
        those two methods and is not a generic alternative to ``barrier()``.
        """
        store_key = "{}:{}".format(
            self.STORE_BASED_BARRIER_PREFIX, self._group_count)
        store.add(store_key, 1)
        logging.info(
            'Added key: {} to store for rank: {}'.format(store_key, rank))

        # Now wait for all workers to check in with the store.
        world_size = self.get_world_size()
        # Use 'add' instead of 'get' since for some store implementations 'add'
        # doesn't work well with 'get'. Ideally the store implementations should
        # be fixed, but for backward compatibility reasons it is risky to change
        # the store implementations. Once, we completely migrate away from these
        # legacy stores, we can use 'get' here instead.
        worker_count = store.add(store_key, 0)
        start = time.time()
        log_time = time.time()
        while worker_count != world_size:
            time.sleep(0.01)
            worker_count = store.add(store_key, 0)

            # Print status periodically to keep track.
            if timedelta(seconds=(time.time() - log_time)) > timedelta(seconds=10):
                logging.info(
                    "Waiting in store based barrier to initialize process group for "
                    "rank: {}, key: {} (world_size={}, worker_count={}, timeout={})".format(
                        rank, store_key, world_size, worker_count, timeout))
                log_time = time.time()

            if timedelta(seconds=(time.time() - start)) > timeout:
                raise RuntimeError(
                    "Timed out initializing process group in store based barrier on "
                    "rank: {}, for key: {} (world_size={}, worker_count={}, timeout={})".format(
                        rank, store_key, world_size, worker_count, timeout))

    def _rank_not_in_group(self, group: ProcessGroup) -> bool:
        """
        Helper that checks if the current process's rank is not in a given group.
        """
        if group is None:
            return False
        return group == self.NON_GROUP_MEMBER

    def _get_group_rank(self, group: ProcessGroup, rank: int) -> int:
        """
        Helper that gets a given group's local rank in the group from a given global
        rank.
        """
        if group is self.WORLD:
            raise RuntimeError("WORLD does not have local rank to global "
                               "rank mapping")
        if group not in self._pg_group_ranks:
            raise RuntimeError("The given group does not exist")
        try:
            group_rank = self._pg_group_ranks[group][rank]
        except KeyError:
            raise RuntimeError(
                f"The global rank {rank} is not part of the group {group}") from None
        return group_rank

    def _get_global_rank(self, group: ProcessGroup, group_rank: int) -> bool:
        """
        Helper that gets a given group's global rank from a given local rank in the
        group.
        """
        if group is self.WORLD:
            raise RuntimeError("self.WORLD does not have local rank to global "
                               "rank mapping")
        group_rank_map = self._pg_group_ranks[group]
        for rank, grp_rank in group_rank_map.items():
            if grp_rank == group_rank:
                return rank
        raise RuntimeError("The group rank is not part of the group")

    def _get_group_size(self, group: ProcessGroup) -> int:
        """
        Helper that gets a given group's world size.
        """
        if group is self.WORLD or group is None:
            default_pg = self._get_default_group()
            return default_pg.size()
        if group not in self._pg_group_ranks:
            raise RuntimeError("The given group does not exist")
        return len(self._pg_group_ranks[group])

    def _check_p2p_op_list(self, p2p_op_list: List) -> None:
        """
        Helper to check that the ``p2p_op_list`` is a list of P2POp instances and
        all ops use the same backend.
        """
        if not isinstance(p2p_op_list, list) or \
                not all(isinstance(p2p_op, P2POp) for p2p_op in p2p_op_list):
            raise RuntimeError("Invalid ``p2p_op_list``. Each op is expected to "
                               "to be of type ``torch.distributed.P2POp``.")

        backend = self.get_backend(p2p_op_list[0].group)
        if not all(backend == self.get_backend(p2p_op.group) for p2p_op in p2p_op_list):
            raise RuntimeError("All groups need to use the same backend.")

    def is_initialized(self) -> bool:
        """
        Checking if the default process group has been initialized
        """
        return self.WORLD is not None

    def _get_default_group(self) -> ProcessGroup:
        """
        Getting the default process group created by init_process_group
        """
        if not self.is_initialized():
            raise RuntimeError("Default process group has not been initialized, "
                               "please make sure to call init_process_group.")
        return self.WORLD

    def _get_default_store(self) -> Store:
        """
        Getting the default store created by init_process_group
        """
        if not self.is_initialized():
            raise RuntimeError("Default process group has not been initialized, "
                               "please make sure to call init_process_group.")
        default_pg = self._get_default_group()
        _, default_store = self._pg_map[default_pg]
        return default_store

    def _update_default_pg(self, pg: ProcessGroup) -> None:
        self.WORLD = pg

    def get_backend(self, group: ProcessGroup = None) -> Backend:
        """
        Returns the backend of the given process group.

        Args:
            group (ProcessGroup, optional): The process group to work on. The
                default is the general main process group. If another specific group
                is specified, the calling process must be part of :attr:`group`.

        Returns:
            The backend of the given process group as a lower case string.

        """
        if group is None:
            pg = self._get_default_group()
        else:
            pg = group
        if self._rank_not_in_group(pg):
            raise RuntimeError("Invalid process group specified")
        pg_store = self._pg_map.get(pg, None)
        assert pg_store is not None
        return pg_store[0]

    def get_store(self, group: ProcessGroup = None) -> Store:
        """
        Returns the store/prefix_store of the given group.

        Args:
            group (ProcessGroup, optional): The process group to work on. The
                default is the general main process group. If another specific group
                is specified, the calling process must be part of :attr:`group`.

        Returns:
            The store/prefix_store of the given process group.
        """
        if group is None:
            pg = self._get_default_group()
        else:
            pg = group
        if self._rank_not_in_group(pg):
            raise RuntimeError("Invalid process group specified")
        pg_store = self._pg_map.get(pg, None)
        assert pg_store is not None
        return pg_store[1]

    def init_process_group(self,
                           backend: Union[str, Backend],
                           init_method: str = None,
                           timeout: timedelta = DEFAULT_PG_TIMEOUT,
                           world_size: int = -1,
                           rank: int = -1,
                           store: Store = None,
                           group_name: str = '') -> None:
        """
        Initializes the default distributed process group, and this will also
        initialize the distributed package.

        There are 2 main ways to initialize a process group:
            1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
            2. Specify ``init_method`` (a URL string) which indicates where/how
            to discover peers. Optionally specify ``rank`` and ``world_size``,
            or encode all required parameters in the URL and omit them.

        If neither is specified, ``init_method`` is assumed to be "env://".


        Args:
            backend (str or Backend): The backend to use. Depending on
                build-time configurations, valid values include ``mpi``, ``gloo``,
                and ``nccl``. This field should be given as a lowercase string
                (e.g., ``"gloo"``), which can also be accessed via
                :class:`Backend` attributes (e.g., ``Backend.GLOO``). If using
                multiple processes per machine with ``nccl`` backend, each process
                must have exclusive access to every GPU it uses, as sharing GPUs
                between processes can result in deadlocks.
            init_method (str, optional): URL specifying how to initialize the
                                        process group. Default is "env://" if no
                                        ``init_method`` or ``store`` is specified.
                                        Mutually exclusive with ``store``.
            world_size (int, optional): Number of processes participating in
                                        the job. Required if ``store`` is specified.
            rank (int, optional): Rank of the current process (it should be a
                                number between 0 and ``world_size``-1).
                                Required if ``store`` is specified.
            store(Store, optional): Key/value store accessible to all workers, used
                                    to exchange connection/address information.
                                    Mutually exclusive with ``init_method``.
            timeout (timedelta, optional): Timeout for operations executed against
                the process group. Default value equals 30 minutes.
                This is applicable for the ``gloo`` backend. For ``nccl``, this is
                applicable only if the environment variable ``NCCL_BLOCKING_WAIT``
                or ``NCCL_ASYNC_ERROR_HANDLING`` is set to 1. When
                ``NCCL_BLOCKING_WAIT`` is set, this is the duration for which the
                process will block and wait for collectives to complete before
                throwing an exception. When ``NCCL_ASYNC_ERROR_HANDLING`` is set,
                this is the duration after which collectives will be aborted
                asynchronously and the process will crash. ``NCCL_BLOCKING_WAIT``
                will provide errors to the user which can be caught and handled,
                but due to its blocking nature, it has a performance overhead. On
                the other hand, ``NCCL_ASYNC_ERROR_HANDLING`` has very little
                performance overhead, but crashes the process on errors. This is
                done since CUDA execution is async and it is no longer safe to
                continue executing user code since failed async NCCL operations
                might result in subsequent CUDA operations running on corrupted
                data. Only one of these two environment variables should be set.
            group_name (str, optional, deprecated): Group name.

        To enable ``backend == Backend.MPI``, PyTorch needs to be built from source
        on a system that supports MPI.

        """
        if not isinstance(timeout, timedelta):
            raise RuntimeError("Expected timeout argument to be of type"
                               "datetime.timedelta")

        if self.WORLD is not None:
            raise RuntimeError("trying to initialize the default process group "
                               "twice!")

        assert (store is None) or (init_method is None), \
            "Cannot specify both init_method and store."

        if store is not None:
            assert world_size > 0, 'world_size must be positive if using store'
            assert rank >= 0, 'rank must be non-negative if using store'
        elif init_method is None:
            init_method = "env://"

        def init_store(rank, world_size, timeout):
            rendezvous_iterator = rendezvous(
                init_method, rank, world_size, timeout=timeout
            )
            store, rank, world_size = next(rendezvous_iterator)
            return store, rank, world_size

        tmp_timeout = DEFAULT_PG_SHORT_TIMEOUT if rank == 0 else DEFAULT_PG_LONG_TIMEOUT

        def attempt_init_store(rank, world_size, timeout):
            store, rank, world_size = init_store(rank, world_size, timeout)
            # Test each other.
            if rank == 0:
                store.set("RANK_ZERO", "True")
                store.set_timeout(timeout)
                store.get("RANK_OTHER")
            else:
                store.set("RANK_OTHER", "True")
                store.set_timeout(timeout)
                store.get("RANK_ZERO")

            return store, rank, world_size

        # whatever the backend is, we need a store to exchange information.
        if store is None:
            try:
                store, rank, world_size = attempt_init_store(
                    rank, world_size, tmp_timeout)
            except Exception as e:
                logger.exception(e)
                raise ConnectTimeout
            finally:
                ...
            store.set_timeout(timeout)

        backend = Backend(backend)

        def init_backend(timeout):
            if backend == Backend.MPI:
                if world_size != -1 or rank != -1:
                    warnings.warn(
                        "For MPI backend, world_size ({}) and rank ({}) "
                        "are ignored since they are assigned by the "
                        "MPI runtime.".format(world_size, rank))

                self._update_default_pg(self._new_process_group_helper(
                    -1,
                    -1,
                    [],
                    Backend.MPI,
                    None,
                    group_name=group_name,
                    timeout=timeout))
            else:
                self._update_default_pg(self._new_process_group_helper(
                    world_size,
                    rank,
                    [],
                    backend,
                    store,
                    group_name=group_name,
                    timeout=timeout))

        init_backend(timeout)

        self._pg_group_ranks[self.WORLD] = {
            i: i for i in range(self.WORLD.size())}  # type: ignore
        self._default_pg_init_method = init_method

        # barrier at the end to ensure that once we return from this method, all
        # process groups including global variables are updated correctly on all
        # ranks.
        if backend == Backend.MPI:
            # MPI backend doesn't use store.
            self.barrier()
        else:
            # Use store based barrier here since barrier() used a bunch of
            # default devices and messes up NCCL internal state.
            self._store_based_barrier(rank, store, timeout)

    def _new_process_group_helper(self,
                                  world_size: int,
                                  rank: int,
                                  group_ranks: int,
                                  backend: Union[str, Backend],
                                  store: Store,
                                  group_name: str = None,
                                  timeout: timedelta = DEFAULT_PG_TIMEOUT) -> ProcessGroup:
        """
        Create a new distributed process group.

        This function must be called by ALL processes in the global group, even if
        the calling process is not part of the newly created group. In that case,
        this function returns self.NON_GROUP_MEMBER.

        This function is called with ``group_ranks == []`` for the default group.
        """
        if not group_name:
            group_name = str(self._group_count)
        self._group_count += 1

        if group_name in self._pg_names.values():
            raise RuntimeError("The specified group name has already been "
                               "created, please use a different group name")

        if not isinstance(timeout, timedelta):
            raise RuntimeError("Expected timeout argument to be of type"
                               "datetime.timedelta")

        # The list of group ranks is empty if we're creating the default group.
        is_default_group = (len(group_ranks) == 0)

        backend = Backend(backend)
        pg: Union[ProcessGroupGloo, ProcessGroupMPI, ProcessGroupNCCL]

        # Use the group name as prefix in the default store, such that
        # a single store can be reused by multiple groups.
        prefix_store = PrefixStore(group_name, store)

        def connect_backend():
            if backend == Backend.MPI:
                if not is_mpi_available():
                    raise RuntimeError(
                        "Distributed package doesn't have MPI built in."
                        " MPI is only included if you build PyTorch from"
                        " source on a host that has MPI installed.")
                pg = ProcessGroupMPI.create(group_ranks)
                if not pg:
                    return self.NON_GROUP_MEMBER
                self._pg_map[pg] = (Backend.MPI, prefix_store)
                self._pg_names[pg] = group_name
            else:
                # If this is a subgroup (which means group_ranks is specified),
                # we check if the current process is a member of the new group.
                if not is_default_group:
                    global_rank = self._get_default_group().rank()
                    if global_rank not in group_ranks:
                        return self.NON_GROUP_MEMBER

                if backend == Backend.GLOO:
                    pg = ProcessGroupGloo(
                        prefix_store,
                        rank,
                        world_size,
                        timeout=timeout)
                    self._pg_map[pg] = (Backend.GLOO, prefix_store)
                    self._pg_names[pg] = group_name
                elif backend == Backend.NCCL:
                    if not is_nccl_available():
                        raise RuntimeError("Distributed package doesn't have NCCL "
                                           "built in")
                    pg = ProcessGroupNCCL(
                        prefix_store,
                        rank,
                        world_size,
                        timeout)
                    self._pg_map[pg] = (Backend.NCCL, prefix_store)
                    self._pg_names[pg] = group_name
                else:
                    pg = getattr(Backend, backend.upper())(
                        prefix_store,
                        rank,
                        world_size,
                        timeout)
                    self._pg_map[pg] = (backend, prefix_store)
                    self._pg_names[pg] = group_name

            return pg

        try:
            acquire_all()
            pg = connect_backend()
        except RuntimeError as re:
            logger.exception(re)
            raise ConnectTimeout(re)
        except TimeoutError as te:
            logger.exception(te)
            raise ConnectTimeout(te)
        except Exception as e:
            raise e
        finally:
            release_all()
        return pg

    def destroy_process_group(self, group: ProcessGroup = None) -> None:
        """
        Destroy a given process group, and reinitialize the distributed package

        Args:
            group (ProcessGroup, optional): The process group to be destroyed, if
                                            self.WORLD is given, all process
                                            groups including the default one will
                                            be destroyed.
        """

        if group == self.NON_GROUP_MEMBER:
            return

        if group is None:
            pg = self.WORLD
        else:
            pg = group

        assert pg is not None
        if self._pg_map.get(pg, None) is None:
            raise RuntimeError("Invalid process group specified")

        if group is None or group == self.WORLD:
            self._update_default_pg(None)
            self._default_pg_init_method = None
            self._pg_map.clear()
            self._point2point_groups.clear()
            self._pg_names.clear()
            self._pg_group_ranks.clear()

            # when process group doesn't have an explicit name (only WORLD (default)
            # process group can have an explicit name), we use global _group_counter
            # to generate the name. We need to reset the counter on destruction to
            # allow consistent value to be generated when we re-create process
            # groups after some trainers recover from failure
            #
            # We only reset this when WORLD is being destroyed because if this
            # process group is in good state, we aren't dealing with failures.
            self._group_count = 0
        else:
            del self._pg_map[pg]
            del self._pg_names[pg]
            del self._pg_group_ranks[pg]

            if group in self._point2point_groups:
                del self._point2point_groups[group]
            self._group_count -= 1

    def get_rank(self, group: ProcessGroup = None) -> int:
        """
        Returns the rank of current process group

        Rank is a unique identifier assigned to each process within a distributed
        process group. They are always consecutive integers ranging from 0 to
        ``world_size``.

        Args:
            group (ProcessGroup, optional): The process group to work on. If None,
                the default process group will be used.

        Returns:
            The rank of the process group
            -1, if not part of the group

        """
        if self._rank_not_in_group(group):
            return -1

        default_pg = self._get_default_group()
        if group is None or group is self.WORLD:
            return default_pg.rank()

        return self._get_group_rank(group, default_pg.rank())

    def get_world_size(self, group: ProcessGroup = None) -> int:
        """
        Returns the number of processes in the current process group

        Args:
            group (ProcessGroup, optional): The process group to work on. If None,
                the default process group will be used.

        Returns:
            The world size of the process group
            -1, if not part of the group

        """
        if self._rank_not_in_group(group):
            return -1

        return self._get_group_size(group)

    def _validate_output_list_for_rank(self, my_rank: int, dst: int, gather_list: List[Any]) -> None:
        if dst == my_rank:
            if not gather_list:
                raise ValueError(
                    "Argument ``gather_list`` must be specified on destination rank."
                )
        elif gather_list:
            raise ValueError(
                "Argument ``gather_list`` must NOT be specified "
                "on non-destination ranks."
            )

    def barrier(self,
                group: ProcessGroup = None,
                async_op: bool = False,
                device_ids: int = None):
        """
        Synchronizes all processes.

        This collective blocks processes until the whole group enters this function,
        if async_op is False, or if async work handle is called on wait().

        Args:
            group (ProcessGroup, optional): The process group to work on. If None,
                the default process group will be used.
            async_op (bool, optional): Whether this op should be an async op
            device_ids ([int], optional): List of device/GPU ids.
                                        Valid only for NCCL backend.

        Returns:
            Async work handle, if async_op is set to True.
            None, if not async_op or if not part of the group
        """
        if group is None:
            group = self.WORLD
        if self._rank_not_in_group(group):
            return

        opts = BarrierOptions()
        if device_ids is not None:
            if self.get_backend(group) != Backend.NCCL:
                raise RuntimeError("Function argument device_ids not supported "
                                   "for the selected backend {}".format(self.get_backend(group)))
            if isinstance(device_ids, list):
                opts.device_ids = device_ids
            else:
                raise RuntimeError("Invalid function argument: "
                                   "device_ids type should be List[int]")

        if group is None:
            default_pg = self._get_default_group()
            work = default_pg.barrier(opts=opts)
        else:
            work = group.barrier(opts=opts)

        if async_op:
            return work
        else:
            work.wait()

    def new_group(self,
                  ranks: int = None,
                  timeout: timedelta = DEFAULT_PG_TIMEOUT,
                  backend: Union[str, Backend] = None,
                  group_name: str = None) -> ProcessGroup:
        """
        Creates a new distributed group.

        This function requires that all processes in the main group (i.e. all
        processes that are part of the distributed job) enter this function, even
        if they are not going to be members of the group. Additionally, groups
        should be created in the same order in all processes.

        .. warning::
            Using multiple process groups with the ``NCCL`` backend concurrently
            is not safe and the user should perform explicit synchronization in
            their application to ensure only one process group is used at a time.
            This means collectives from one process group should have completed
            execution on the device (not just enqueued since CUDA execution is
            async) before collectives from another process group are enqueued.
            See `Using multiple NCCL communicators concurrently <https://docs.nvid
            ia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using
            -multiple-nccl-communicators-concurrently>`_ for more details.

        Args:
            ranks (list[int]): List of ranks of group members. If ``None``, will be
                set to all ranks. Default is ``None``.
            timeout (timedelta, optional): Timeout for operations executed against
                the process group. Default value equals 30 minutes.
                This is only applicable for the ``gloo`` backend.
            backend (str or Backend, optional): The backend to use. Depending on
                build-time configurations, valid values are ``gloo`` and ``nccl``.
                By default uses the same backend as the global group. This field
                should be given as a lowercase string (e.g., ``"gloo"``), which can
                also be accessed via :class:`Backend` attributes (e.g.,
                ``Backend.GLOO``).

        Returns:
            A handle of distributed group that can be given to collective calls.
        """

        default_pg = self._get_default_group()
        default_backend, default_store = self._pg_map[default_pg]
        global_rank = default_pg.rank()
        global_world_size = default_pg.size()

        # Default to the same backend as the global process group
        # if the backend is not specified.
        if not backend:
            backend = default_backend

        # checks the input ranks
        if ranks is not None:
            ranks = sorted(ranks)
            group_world_size = len(ranks)
            if group_world_size > global_world_size:
                raise RuntimeError("the new group's world size should be less or "
                                   "equal to the world size set by "
                                   "init_process_group")
            # check ranks' sanity
            for rank in ranks:
                if rank < 0 or rank >= global_world_size:
                    raise RuntimeError("The new group's rank should be within the "
                                       "the world_size set by init_process_group")
            if global_rank in ranks:
                group_rank = ranks.index(global_rank)
            else:
                group_rank = None
        else:
            ranks = list(range(global_world_size))
            group_world_size = global_world_size
            group_rank = global_rank

        backend = Backend(backend)
        pg = self._new_process_group_helper(group_world_size,
                                            group_rank,
                                            ranks,
                                            backend,
                                            default_store,
                                            timeout=timeout,
                                            group_name=group_name,)

        # Create the global rank to group rank mapping
        self._pg_group_ranks[pg] = {
            global_rank: group_rank
            for group_rank, global_rank in enumerate(ranks)
        }

        # barrier at the end to ensure that once we return from this method, all
        # process groups including global variables are updated correctly on all
        # ranks.
        if backend == Backend.MPI:
            # MPI doesn't have store.
            self.barrier()
        else:
            # Use store based barrier here since barrier() used a bunch of
            # default devices and messes up NCCL internal state.
            self._store_based_barrier(global_rank, default_store, timeout)

        return pg

    def build_point2point_group(self,
                                rank: int = 0,
                                timeout: timedelta = DEFAULT_PG_TIMEOUT,
                                backend: Union[str, Backend] = None) -> List[ProcessGroup]:
        """Build point2point group, :param:rank will be regarded as new rank=0 and connect to other rank in this world.

        .. note::
            Only build this if you really need it. Otherwise, please use new_group() to build single one.
        """
        assert 0 <= rank < self.get_world_size()
        pg_list = []
        for other in range(self.get_world_size()):
            if other == rank:
                # skip self to self connection
                continue
            else:
                pg = self.new_group(
                    ranks=[rank, other], timeout=timeout, backend=backend, group_name=f"point2point-{rank}-{other}",)
                # backup
                if pg is not self.NON_GROUP_MEMBER:
                    self._point2point_groups[pg] = self._pg_map[pg]
                    pg_list.append(pg)
        return pg_list

    def __repr__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Country",
            description=f"Belongs to\n{self.world}"
        )
