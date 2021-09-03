# Copyright (c) FederalLab. All rights reserved.
import cmd
import glob
import os

from openfed.common import Address
from openfed.topo import Edge, Node, Topology
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
