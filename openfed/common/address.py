import json
import os
from argparse import Namespace
from typing import List, TypeVar, overload

from openfed.utils import openfed_class_fmt
from openfed.utils.table import tablist

_A = TypeVar("_A", bound='Address')


class Address(object):
    backend: str
    init_method: str = None
    world_size: int = 2
    rank: int = -1
    store = None
    group_name: str = ''

    @overload
    def __init__(self, args: Namespace):
        """Load address from parser.
        """

    @overload
    def __init__(self,
                 backend: str,
                 init_method: str = None,
                 world_size: int = 2,
                 rank: int = -1,
                 store=None,
                 group_name: str = ''):
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

    def __init__(self, **kwargs):
        if kwargs.get('args', None):
            args = kwargs.get('args')
            if args.port is not None:
                if args.init_method.startswith("tcp"):
                    args.init_method = ":".join(
                        args.init_method.split(":")[:2] + [str(args.port)])

            self.backend = args.backend
            self.init_method = args.init_method
            self.world_size = args.world_size
            self.rank = args.rank
            self.store = None
            self.group_name = args.group_name
        else:
            self.backend = kwargs['backend']
            self.init_method = kwargs.get('init_method', None)
            self.world_size = kwargs.get('world_size', 2)
            self.rank = kwargs.get('rank', -1)
            self.store = kwargs.get('store', None)
            self.group_name = kwargs.get('group_name', "")

    @classmethod
    def load_from_file(cls, file: str) -> List[_A]:
        if not os.path.isfile(file):
            return []
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
        table = tablist(
            head=['Backend', 'Init Method', 'World Size',
                  'Rank', 'Store', 'Group Name'],
            data=[self.backend, self.init_method, self.world_size,
                  self.rank, self.store, self.group_name],
            force_in_one_row=True
        )
        return openfed_class_fmt.format(
            class_name="Address",
            description=table,
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
