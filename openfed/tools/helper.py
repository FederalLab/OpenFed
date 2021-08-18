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
r"""
`openfed.tools.helper` is a module that assists you better manage federated address file.

**How to use this module:**

::

    >>> python -m openfed.tools.helper
        A helper script to manage json file of federated address for <OpenFed>.
    >>> <OpenFed> /Users/densechen/code/OpenFed: help

        Documented commands (type help <topic>):
        ========================================
        add  cd  conf  del  exit  help  ls  read  save

    >>> <OpenFed> /Users/densechen/code/OpenFed: conf openfed_addresses
        File not exists: openfed_addresses.json
        Create a new one.
    >>> <OpenFed> *openfed_addresses.json*: add
        Backend (gloo, mpi, nccl): gloo
        Init method (tcp://IP:PORT, file://PATH_TO_FILE, env://): tcp://192.168.0.1:1994
        Group name: openfed
        World size: 30
        Rank: 0
        Add a new address.
    >>> <OpenFed> Address
        +---------+---------------+------------+------+-------+------------+
        | backend |  init_method  | world_size | rank | store | group_name |
        +---------+---------------+------------+------+-------+------------+
        |   gloo  | tcp://192.... |     30     |  0   |  None |  openfed   |
        +---------+---------------+------------+------+-------+------------+

    >>> <OpenFed> *openfed_addresses.json*: add
        Backend (gloo, mpi, nccl): nccl
        Init method (tcp://IP:PORT, file://PATH_TO_FILE, env://): file:///tmp/openfed.sharedfile
        Group name: openfed
        World size: 40
        Rank: 39
        Add a new address.
        <OpenFed> Address
        +---------+---------------+------------+------+-------+------------+
        | backend |  init_method  | world_size | rank | store | group_name |
        +---------+---------------+------------+------+-------+------------+
        |   nccl  | file:///tm... |     40     |  39  |  None |  openfed   |
        +---------+---------------+------------+------+-------+------------+

    >>> <OpenFed> *openfed_addresses.json*: read
        Items: 0/2
        <OpenFed> Address
        +---------+---------------+------------+------+-------+------------+
        | backend |  init_method  | world_size | rank | store | group_name |
        +---------+---------------+------------+------+-------+------------+
        |   gloo  | tcp://192.... |     30     |  0   |  None |  openfed   |
        +---------+---------------+------------+------+-------+------------+

        Items: 1/2
        <OpenFed> Address
        +---------+---------------+------------+------+-------+------------+
        | backend |  init_method  | world_size | rank | store | group_name |
        +---------+---------------+------------+------+-------+------------+
        |   nccl  | file:///tm... |     40     |  39  |  None |  openfed   |
        +---------+---------------+------------+------+-------+------------+

    >>> <OpenFed> *openfed_addresses.json*: save
        Items: 0/2
        <OpenFed> Address
        +---------+---------------+------------+------+-------+------------+
        | backend |  init_method  | world_size | rank | store | group_name |
        +---------+---------------+------------+------+-------+------------+
        |   gloo  | tcp://192.... |     30     |  0   |  None |  openfed   |
        +---------+---------------+------------+------+-------+------------+

        Items: 1/2
        <OpenFed> Address
        +---------+---------------+------------+------+-------+------------+
        | backend |  init_method  | world_size | rank | store | group_name |
        +---------+---------------+------------+------+-------+------------+
        |   nccl  | file:///tm... |     40     |  39  |  None |  openfed   |
        +---------+---------------+------------+------+-------+------------+

        Saved address to openfed_addresses.json.
    >>> <OpenFed> *openfed_addresses.json*: del 0
        Delete item 0
        <OpenFed> Address
        +---------+---------------+------------+------+-------+------------+
        | backend |  init_method  | world_size | rank | store | group_name |
        +---------+---------------+------------+------+-------+------------+
        |   gloo  | tcp://192.... |     30     |  0   |  None |  openfed   |
        +---------+---------------+------------+------+-------+------------+

        <OpenFed> *openfed_addresses.json*: save
        Items: 0/1
        <OpenFed> Address
        +---------+---------------+------------+------+-------+------------+
        | backend |  init_method  | world_size | rank | store | group_name |
        +---------+---------------+------------+------+-------+------------+
        |   nccl  | file:///tm... |     40     |  39  |  None |  openfed   |
        +---------+---------------+------------+------+-------+------------+

        Saved address to openfed_addresses.json.
    >>> <OpenFed> *openfed_addresses.json*: read
        Items: 0/1
        <OpenFed> Address
        +---------+---------------+------------+------+-------+------------+
        | backend |  init_method  | world_size | rank | store | group_name |
        +---------+---------------+------------+------+-------+------------+
        |   nccl  | file:///tm... |     40     |  39  |  None |  openfed   |
        +---------+---------------+------------+------+-------+------------+

    >>> <OpenFed> *openfed_addresses.json*: exit
        Items: 0/1
        <OpenFed> Address
        +---------+---------------+------------+------+-------+------------+
        | backend |  init_method  | world_size | rank | store | group_name |
        +---------+---------------+------------+------+-------+------------+
        |   nccl  | file:///tm... |     40     |  39  |  None |  openfed   |
        +---------+---------------+------------+------+-------+------------+

        Saved address to openfed_addresses.json.
    >>> <OpenFed> /Users/densechen/code/OpenFed: ls
        Current directory: /Users/densechen/code/OpenFed
        [0]: /Users/densechen/code/OpenFed/openfed_addresses.json
        1 config files have listed.
    >>> <OpenFed> /Users/densechen/code/OpenFed: exit
    >>> cat openfed_addresses.json
        [{"backend": "nccl", "init_method": "file:///tmp/openfed.sharedfile", "world_size": 40, "rank": 39, "store": null, "group_name": "openfed"}]%
"""

import cmd
import glob
import os
from typing import List, Any

from openfed.common import (Address, dump_address_to_file,
                            load_address_from_file, InvalidAddress)
from openfed.utils import openfed_title


class Helper(cmd.Cmd):
    intro = f"A helper script to manage json file of federated address for {openfed_title}."

    @property
    def prompt(self):
        if self.config_file is not None:
            return f"{openfed_title} *\033[0;32m{self.config_file}\033[0m*: "
        else:
            return f'{openfed_title} {os.getcwd()}: '

    def __init__(self):
        super().__init__()
        self.config_file = None
        self.address_list = []

    # Config file manage
    def do_conf(self, args):
        """Specify a configure file to used.
        It not exists, it will be created automatically.
        """
        if not args.endswith(".json"):
            args = args + ".json"
        if not os.path.exists(args):
            print("File not exists: %s" % args)
            print("Create a new one.")
        self.config_file = args

        self.address_list = load_address_from_file(self.config_file)

    def do_ls(self, args):
        """List all config files in current directory.
        """
        pwd = os.getcwd()
        print("Current directory: %s" % pwd)
        config_file = glob.glob(os.path.join(pwd, "*.json"))
        for i, cf in enumerate(config_file):
            print(f"[{i}]: {cf}")
        else:
            print(f"{len(config_file)} config files have listed.")

    def do_cd(self, args):
        """Cd other directory.
        """
        args = self.parseline(args)
        os.chdir(args[1])  # type: ignore

    def do_exit(self, args):
        """Write file and exit.
        """
        if self.config_file is not None:
            self.do_save()
            self.config_file = None
            self.address_list = []
        else:
            exit(0)

    # Address Manage.

    def do_save(self, *args):
        """Save config files.
        """
        self.do_read()
        if self.config_file is not None:
            dump_address_to_file(self.config_file, self.address_list)
            print(f"Saved address to {self.config_file}.")

    def do_read(self, *args):
        """Print the address list to screen.
        """
        if len(self.address_list) == 0:
            print("No items.")
        else:
            for i, add in enumerate(self.address_list):
                print(f"Items: {i}/{len(self.address_list)}")
                print(str(add))

    def do_del(self, args):
        """Delete a address from list.
        """
        args = int(args)
        print(f"Delete item {args}")
        print(str(self.address_list[args]))

        del self.address_list[args]

    def do_add(self, args):
        """Add a new address to json file.
        """
        backend = input("Backend (gloo, mpi, nccl): ")
        init_method = input(
            "Init method (tcp://IP:PORT, file://PATH_TO_FILE, env://): ")
        if not init_method.startswith("env://"):
            group_name = input("Group name: ")
            world_size = int(input("World size: "))
            rank = int(input("Rank: "))
        else:
            group_name = ''
            world_size = 0
            rank = -1

        try:
            address = Address(backend=backend,
                              group_name=group_name,
                              init_method=init_method,
                              world_size=world_size,
                              rank=rank)
        except InvalidAddress as e:
            print(e)
            return

        # check conflict
        if address in self.address_list:
            print("Already existing address.")
            return
        print("Add a new address.")
        print(address)

        self.address_list.append(address)


if __name__ == '__main__':
    helper = Helper()
    helper.cmdloop()
