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


import cmd
import glob
import os

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
    def do_conf(self, arg):
        """Specify a configure file to used.
        It not exists, it will be created automatically.
        """
        if not arg.endswith(".json"):
            arg = arg + ".json"
        if not os.path.exists(arg):
            print("File not exists: %s" % arg)
            print("Create a new one.")
        self.config_file = arg

        self.address_list = load_address_from_file(self.config_file)

    def do_ls(self, arg):
        """List all config files in current directory.
        """
        pwd = os.getcwd()
        print("Current directory: %s" % pwd)
        config_file = glob.glob(os.path.join(pwd, "*.json"))
        for i, cf in enumerate(config_file):
            print(f"[{i}]: {cf}")
        else:
            print(f"{len(config_file)} config files have listed.")

    def do_cd(self, arg):
        """Cd other directory.
        """
        arg = self.parseline(arg)
        os.chdir(arg[1])

    def do_exit(self, arg):
        """Write file and exit.
        """
        if self.config_file is not None:
            self.do_save()
            self.config_file = None
            self.address_list = []
        else:
            exit(0)

    # Address Manage.

    def do_save(self, *arg):
        """Save config files.
        """
        self.do_read()
        if self.config_file is not None:
            dump_address_to_file(self.config_file, self.address_list)
            print(f"Saved address to {self.config_file}.")

    def do_read(self, *arg):
        """Print the address list to screen.
        """
        if len(self.address_list) == 0:
            print("No items.")
        else:
            for i, add in enumerate(self.address_list):
                print(f"Items: {i}/{len(self.address_list)}")
                print(str(add))

    def do_del(self, arg):
        """Delete a address from list.
        """
        arg = int(arg)
        print(f"Delete item {arg}")
        print(str(self.address_list[arg]))

        del self.address_list[arg]

    def do_add(self, arg):
        """Add a new address to json file.
        """
        backend = input("Backend (gloo, mpi, nccl): ")
        init_method = input(
            "Init method (tcp://IP:PORT, file://PATH_TO_FILE, env://): ")
        group_name = input("Group name: ")
        world_size = int(input("World size: "))
        rank = int(input("Rank: "))

        try:
            address = Address(
                backend=backend, group_name=group_name,
                init_method=init_method, world_size=world_size, rank=rank)
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
