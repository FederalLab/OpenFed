import json
from typing import List, TypeVar

from openfed.utils import openfed_class_fmt
from prettytable import PrettyTable

_A = TypeVar("_A", bound='Address')


class Address(object):
    backend: str
    init_method: str = None
    world_size: int = 2
    rank: int = -1
    store = None
    group_name: str = ''

    def __init__(self,
                 backend: str,
                 init_method: str = None,
                 world_size: int = 2,
                 rank: int = -1,
                 store=None,
                 group_name: str = ''):
        """
        rank设置成-1的时候，会根据当前的身份自动推断。
        仅仅当你是在建立点对点的连接的时候，才会有效。也就是world size==2的时候。

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
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        self.store = store
        self.group_name = group_name

    @classmethod
    def load_from_file(cls, file: str) -> List[_A]:
        with open(file, 'r') as f:
            address_dict_list = json.load(f)
        address_list = [Address(**address) for address in address_dict_list]
        return address_list

    @classmethod
    def dump_to_file(cls, file: str, address_list: List[_A]):
        address_dict_list = [address.as_dict for address in address_list]
        with open(file, "w") as f:
            json.dump(address_dict_list, f)

    def __repr__(self):
        # Return a short description
        return openfed_class_fmt.format(
            class_name="Address",
            description=f"@ {self.group_name}",
        )

    def __str__(self):
        # return a long description
        table = PrettyTable(
            ['Backend', 'Init Method', 'World Size', 'Rank', 'Store', 'Group Name']
        )
        table.add_row(
            [self.backend, self.init_method, self.world_size,
                self.rank, self.store, self.group_name]
        )
        return openfed_class_fmt.format(
            class_name="Address",
            description=str(table),
        )

    @property
    def as_dict(self):
        return dict(
            backend=self.backend,
            init_method=self.init_method,
            world_size=self.world_size,
            rank=self.rank,
            store=self.store,
            group_name=self.group_name
        )


default_address = Address(backend="gloo",
                          init_method='tcp://localhost:1993',
                          group_name="OpenFed"
                          )

default_address_lists = [
    Address(
        backend="gloo",
        init_method='tcp://localhost:1994',
        group_name="Admirable"
    ),
    Address(
        backend="gloo",
        init_method='tcp://localhost:1995',
        group_name="Amazing"
    ),
    Address(
        backend="gloo",
        init_method='tcp://localhost:1996',
        group_name="Astonishing"
    ),
    Address(
        backend="gloo",
        init_method='tcp://localhost:1997',
        group_name="Brilliant"
    ),
]
