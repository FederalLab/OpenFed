# Copyright (c) FederalLab. All rights reserved.
import warnings
from typing import Any, Dict

from openfed.utils import openfed_class_fmt, tablist


class Address(object):
    r'''Contains `backend`, `init_method`, `world_size` and `rank` message to
    build the connection between different federated groups.

    .. warning ::
        ``env://`` is not allowed to be used to avoid conflicts with
        distributed learning in `PyTorch`.

    Args:
        backend: The backend to used. Depending on build-time configurations,
            valid values include ``mpi``, ``gloo`` and  ``nccl``, which depends
            on the `PyTorch` you installed. This field should be given as a
            lowercase string (e.g., ``gloo``), which can also be accessed via
            :class:`Backend` attributes (e.g., ``Backend.GLOO``). ``nccl`` is
            not recommended currently, for that we will always move the tensor
            to `cpu` first before sending to other nodes to avoiding device
            disalignment between `cpu` and `gpu`. Thus, ``nccl`` will not speed
            up the communication phase in `OpenFed`. Default: ``'gloo'``.
        init_method: URL specifying how to initialize the federated group. Such
            as: ``tcp://localhost:1994``, ``file:///tmp/sharefile``. If you use
            ``file://``, make sure the file is not existing.
            Default: ``'tcp://localhost:1994'``
        world_size: Number of nodes in federated group. Default: ``2``
        rank: Rank of current node (it should be a number between 0 and
            ``world_size``-1). If `-1` is provided, rank will be specified
            during runtime. Default: -1

    Examples::

        >>> Address('gloo', 'tcp://localhost:1994', world_size=2, rank=-1)
        <OpenFed> Address
        +---------+---------------------+------------+------+
        | backend |     init_method     | world_size | rank |
        +---------+---------------------+------------+------+
        |   gloo  | tcp://localhost:... |     2      |  -1  |
        +---------+---------------------+------------+------+

        >>> Address('mpi', 'file:///tmp/openfed', world_size=10, rank=0)
        <OpenFed> Address
        +---------+---------------------+------------+------+
        | backend |     init_method     | world_size | rank |
        +---------+---------------------+------------+------+
        |   mpi   | file:///tmp/open... |     10     |  0   |
        +---------+---------------------+------------+------+
    '''

    backend: str
    init_method: str
    world_size: int
    rank: int

    def __init__(self,
                 backend: str = 'gloo',
                 init_method: str = 'tcp://localhost:1994',
                 world_size: int = 2,
                 rank: int = -1):
        assert init_method.startswith('file://') or init_method.startswith(
            'tcp://') or init_method.startswith('null')

        if backend == 'nccl':
            # `nccl` backend can largely speed up the directly communication
            # between two GPUs. Currently, OpenFed is based on
            # `gather_object()` to transfer any tensor between two nodes.
            # The gather_object() function will first cast object to cpu tensor
            # and then move it to the original GPU directly.
            # This will cause device mis-alignment when two nodes
            # using different GPU ids.
            # Considering that Federated Learning mostly deals
            # with communication among different devices,
            # such as cpu and gpu, we are not planning
            # to fix it.
            # You can specify `nccl` backend currently, but it may not
            # bring much different with `gloo`.

            warnings.warn('nccl backend is used.')

        assert backend in ['gloo', 'mpi', 'nccl', 'null']
        assert 1 <= world_size
        assert -1 <= rank < world_size

        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank

    def __repr__(self):
        head = ['backend', 'init_method', 'world_size', 'rank']
        data = [self.backend, self.init_method, self.world_size, self.rank]
        description = tablist(head, data, force_in_one_row=True)

        return openfed_class_fmt.format(
            class_name=self.__class__.__name__, description=description)

    def __eq__(self, other):
        return self.backend == other.backend \
            and self.init_method == other.init_method

    def serialize(self) -> Dict[str, Any]:
        return dict(
            backend=self.backend,
            init_method=self.init_method,
            world_size=self.world_size,
            rank=self.rank,
        )

    @classmethod
    def unserialize(cls, data: Dict[str, Any]):
        return Address(**data)


default_tcp_address = Address(
    backend='gloo',
    init_method='tcp://localhost:1994',
)

default_file_address = Address(
    backend='gloo',
    init_method='file:///tmp/openfed.sharedfile',
)

empty_address = Address(
    backend='null',
    init_method='null',
)
