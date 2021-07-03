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

from openfed.common import Address, parser


class Helper(cmd.Cmd):
    intro = "A helper script to manage address for \033[0;34m<OpenFed>\033[0m."

    @property
    def prompt(self):
        if self.config_file is not None:
            return f"\033[0;34m<OpenFed>\033[0m *\033[0;32m{self.config_file}\033[0m*: "
        else:
            return f'\033[0;34m<OpenFed>\033[0m {os.getcwd()}: '

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

        self.address_list = Address.load_from_file(self.config_file)

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
            Address.dump_to_file(self.config_file, self.address_list)
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
        """Add a new address. Currently, env:// is not supported.
        Example:
            add --backend "gloo" --init_method 'tcp://localhost:1994' --group_name "Admirable" --world_size 2 --rank -1 [--port 12345]
        If port is specified, then init_method will be replaced with new port.
        """
        arg = self.parseline(arg)[1]
        if arg:
            namespace = parser.parse_known_args(arg.split(" "))[0]
        else:
            namespace = parser.parse_args()
        address = Address(args=namespace)
        print(str(address))

        # check conflict
        for add in self.address_list:
            if str(add) == str(address):
                print("Already exists.")
                return
        print("Add a new address.")

        self.address_list.append(address)


if __name__ == '__main__':
    helper = Helper()
    helper.cmdloop()
