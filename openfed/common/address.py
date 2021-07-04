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
import time
from argparse import Namespace
from typing import Any, Dict, List, TypeVar, Union, overload

from openfed.utils import convert_to_list, openfed_class_fmt, tablist

from .base import InvalidAddress, peeper

_A = TypeVar("_A", bound='Address_')


# address -> create time
peeper.address_pool = dict()


def cmp_address(add_a: _A, add_b: _A) -> bool:
    if add_a == add_b:
        return True
    else:
        if add_a is None or add_b is None:
            return False
        else:
            for k in add_a.address.keys():
                if add_a.address[k] != add_b.address[k]:
                    return False
            else:
                return True


def add_address_to_pool(address: _A) -> _A:
    # Check address is in pool or not
    for add in peeper.address_pool:
        if cmp_address(address, add):
            del address
            return add
    else:
        peeper.address_pool[address] = time.time()
        return address


def remove_address_from_pool(address: _A) -> bool:
    if address in peeper.address_pool:
        del peeper.address_pool[address]
        return True
    else:
        return False


class Address_(object):

    address: Dict[str, Any] = None

    def __init__(self, **kwargs):
        if kwargs.get('args', None):
            args = kwargs.get('args')
            backend = args.fed_backend
            init_method = args.fed_init_method
            world_size = args.fed_world_size
            rank = args.fed_rank
            store = None
            group_name = args.fed_group_name
        else:
            backend = kwargs['backend']
            init_method = kwargs.get('init_method', 'env://')
            world_size = kwargs.get('world_size', 2)
            rank = kwargs.get('rank', -1)
            store = kwargs.get('store', None)
            group_name = kwargs.get('group_name', '')
        if init_method.startswith('env://'):
            try:
                rank = int(os.environ['FED_RANK'])
                world_size = int(os.environ['FED_WORLD_SIZE'])
                group_name = os.environ['FED_GROUP_NAME']
                # Rename
                # In backend, it will read the value with `FED` prefix.
                # So, rename it.
                os.environ['RANK'] = os.environ['FED_RANK']
                os.environ['LOCAL_RANK'] = os.environ['FED_LOCAL_RANK']
                os.environ['WORLD_SIZE'] = os.environ['FED_WORLD_SIZE']
                os.environ['GROUP_NAME'] = os.environ['FED_GROUP_NAME']
            except KeyError as e:
                raise InvalidAddress(e)

        if backend not in ['gloo', 'mpi', 'nccl']:
            raise InvalidAddress(
                'backend must be one of `gloo`, `mpi`, `nccl`')
        if not (init_method.startswith('file://') or init_method.startswith('env://') or init_method.startswith('tcp://')):
            raise InvalidAddress(
                'init method must start with `file://`, `env://`, `tcp://`')
        if not (-1 <= rank < world_size):
            raise InvalidAddress(
                f"Rank out of index. (rank={rank}, world_size={world_size})")

        self.address = dict(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            store=store,
            group_name=group_name
        )

    def __str__(self) -> str:
        return openfed_class_fmt.format(
            class_name="Address",
            description=tablist(
                head=self.address.keys(),
                data=self.address.values(),
                force_in_one_row=True,
            ),
        )


def load_address_from_file(file: str) -> List[_A]:
    if file is None or not os.path.isfile(file):
        return []
    with open(file, 'r') as f:
        address_dict_list = json.load(f)
    address_list = [Address(**address) for address in address_dict_list]
    return address_list


def dump_address_to_file(file: str, address_list: Union[_A, List[_A]]):
    address_list = convert_to_list(address_list)
    address_dict_list = [address.address for address in address_list]
    with open(file, "w") as f:
        json.dump(address_dict_list, f)


@overload
def Address(args: Namespace):
    """Load address from parser.
    """


@overload
def Address(backend: str,
            init_method: str = "env://",
            world_size: int = 2,
            rank: int = -1,
            store=None,
            group_name: str = ''):
    """
    Initializes the default federated process group.

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
                                    If "env://" is specified, FED_GROUP_NAME,
                                    FED_RANK, FED_WORLD_SIZE should be specified
                                    in the environments. Such as: export FED_RANK=0...
        world_size (int, optional): Number of processes participating in
                                    the job. Required if ``store`` is specified.
        rank (int, optional): Rank of the current process (it should be a
                            number between 0 and ``world_size``-1).
                            Required if ``store`` is specified.
        store(Store, optional): Key/value store accessible to all workers, used
                                to exchange connection/address information.
                                Mutually exclusive with ``init_method``.
        group_name (str, optional): Group name, this name will help you better
            recognize different federated addresses.

    To enable ``backend == Backend.MPI``, PyTorch needs to be built from source
    on a system that supports MPI.
    """


def Address(*args, **kwargs) -> _A:
    address = Address_(*args, **kwargs)
    return add_address_to_pool(address)


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
