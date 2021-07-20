# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import json
import os
import warnings
from collections import namedtuple
from typing import Any, List, Union

from openfed.utils import convert_to_list

from .exceptions import InvalidAddress

Address = namedtuple('Address',
    field_names=['backend', 'init_method', 'world_size', 'rank', 'store', 'group_name'],
    defaults=['gloo', 'env://', 2, -1, None, 'openfed'])


def build_address(
    backend    : str,
    init_method: str,
    world_size : int = 2,
    rank       : int = -1,
    store      : Any = None,
    group_name : str = 'openfed') -> Address: 
    """
    Build a federated address to initialize federated process group.

    There are 2 main ways to initialize a process group:
        1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
        2. Specify ``init_method`` (a URL string) which indicates where/how
        to discover peers. Optionally specify ``rank`` and ``world_size``,
        or encode all required parameters in the URL and omit them.

    .. warn::
        Currently, the first way (using store) is not allowed.

    Args:
        backend: The backend to use. Depending on
            build-time configurations, valid values include ``mpi``, ``gloo``,
            and ``nccl``. This field should be given as a lowercase string
            (e.g., ``gloo``), which can also be accessed via
            :class:`Backend` attributes (e.g., ``Backend.GLOO``). ``nccl`` 
            is not recommended currently.
        init_method: URL specifying how to initialize the
            process group. Such as: ``tcp://localhost:1994``, ``file:///tmp/sharefile``.
            If you use ``file://``, the share file must not be existing.
        world_size: Number of processes participating in the job. 
        rank: Rank of the current process (it should be a
            number between 0 and ``world_size``-1). If -1 is provided, it will
            be specified at runtime (only when ``world_size==2``).
        store: Key/value store accessible to all workers, used
            to exchange connection/address information. Mutually exclusive with 
            ``init_method``.
        group_name: Group name, this name will help you better
            recognize different federated addresses.

    Returns:
        Address.

    To enable ``backend == Backend.MPI``, PyTorch needs to be built from source
    on a system that supports MPI.
    """
    if init_method.startswith('env://'):
        # env:// method will read rank and world size from environment directly.
        # At the same time, it needs a store to exchange information.
        # In OpenFed, ``store`` is not used at most cases.
        # What's more, it is not a good idea to modify any environment variables.
        # Therefore, we disable it.
        raise InvalidAddress('"env://" init method is not allowed in openfed.')

    assert init_method.startswith(
        'file://') or init_method.startswith('tcp://')

    if backend == 'nccl':
        # `nccl` backend can largely speed up the directly communication between
        # two GPUs. Currently, OpenFed is based on `gather_object()` function to
        # transfer any tensor between two processes. The gather_object() function
        # will first cast object to cpu tensor and then move it to GPU. After
        # transfer it to target process, it will restore as the original GPU tensor.
        # This is not ideal features and will harm the final performance.
        # Only when we discard the `gather_object` function, but use related tensor
        # directly function to do communication, can this bottleneck be avoided.
        # However, if we discard the `gather_object` function, a large portion of code
        # needed be rewritten.
        # Considering that Federated Learning mostly deals with variable device, such as
        # cpu vs. gpu, cpu vs. cpu, gpu vs. gpu, and many uncertainly link connection,
        # we are not going to optimize this.
        # Currently, you can specify `nccl` backend, but it may not bring a significant
        # performance improvements.
        raise InvalidAddress('"nccl" backend is not supported currently.')

    assert backend in ['gloo', 'mpi']

    if rank == -1 and world_size == 2:
        # If rank is not specified and world_size is 2, this is a standard point to point
        # connection between leader and follower. We will re-arange the rank for the role.
        # Even though the rank is not -1, it will be forced to modify.
        warnings.warn(
            "Rank will be automatically determined depending on the `role` it plays.")
    else:
        assert 0 <= rank < world_size

    return Address(
        backend     = backend,
        init_method = init_method,
        world_size  = world_size,
        rank        = rank,
        store       = store,
        group_name  = group_name
    )


def load_address_from_file(file: Union[None, str]) -> List[Address]:
    """Load address from file.
    Args:
        file: address json file generated by `openfed.tools.helper`.
    """
    if file is None or not os.path.isfile(file):
        return []
    with open(file, 'r') as f:
        address_dict_list = json.load(f)
    return [Address(*address) for address in address_dict_list]


def dump_address_to_file(file: str, address_list: List[Address]):
    """Dump address to file.
    Args:
        file: file name.
        address_list: list of address.
    """
    address_list = convert_to_list(address_list)
    with open(file, "w") as f:
        json.dump(address_list, f)


default_tcp_address = Address(
    backend     = "gloo",
    init_method = 'tcp://localhost:1994',
    group_name  = "OpenFed"
)

default_file_address = Address(
    backend     = "gloo",
    init_method = 'file:///tmp/openfed.sharedfile',
    group_name  = "OpenFed"
)
