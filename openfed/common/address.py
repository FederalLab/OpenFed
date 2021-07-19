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


import argparse
import json
import os
from argparse import Namespace
from collections import namedtuple
from typing import Any, List, Union, overload

from openfed.utils import convert_to_list

from .base import InvalidAddress

Address = namedtuple('Address',
                     field_names=['backend', 'init_method',
                                  'world_size', 'rank', 'store', 'group_name'],
                     defaults=['gloo', 'env://', 2, -1, None, 'openfed'
                               ])




parser = argparse.ArgumentParser("OpenFed")

# Add parser to address
parser.add_argument(
    "--fed_backend",
    default="gloo",
    type=str,
    choices=["gloo", "mpi", "nccl"], )
parser.add_argument(
    "--fed_init_method",
    default="tcp://localhost:1994",
    type=str,
    help="opt1: tcp://IP:PORT, opt2: file://PATH_TO_SHAREFILE, opt3:env://")
parser.add_argument(
    "--fed_world_size",
    default=2,
    type=int)
parser.add_argument(
    "--fed_rank",
    "--fed_local_rank",
    default=-1,
    type=int)
parser.add_argument(
    "--fed_group_name",
    default="Admirable",
    type=str,
    help="Add a group name to better recognize each address.")

@overload
def build_address(args: Namespace) -> Address:
    """Load address from parser.
    """

@overload
def build_address(backend: str,
                  init_method: str = "env://",
                  world_size: int = 2,
                  rank: int = -1,
                  store: Any = None,
                  group_name: str = '') -> Address:
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


def build_address(*args, **kwargs) -> Address:
    if kwargs.get('args', None):
        args = kwargs.get('args')
        backend = args.fed_backend  # type: ignore
        init_method = args.fed_init_method  # type: ignore
        world_size = args.fed_world_size  # type: ignore
        rank = args.fed_rank  # type: ignore
        store = None
        group_name = args.fed_group_name  # type: ignore
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

    return Address(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        store=store,
        group_name=group_name
    )


def load_address_from_file(file: Union[None, str]) -> List[Address]:
    if file is None or not os.path.isfile(file):
        return []
    with open(file, 'r') as f:
        address_dict_list = json.load(f)
    return [Address(*address) for address in address_dict_list]


def dump_address_to_file(file: str, address_list: List[Address]):
    address_list = convert_to_list(address_list)
    with open(file, "w") as f:
        json.dump(address_list, f)


default_address = Address(
    backend="gloo",
    init_method='tcp://localhost:1993',
    group_name="OpenFed")
