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

parser = argparse.ArgumentParser("OpenFed")

# Add parser to address
parser.add_argument(
    "--backend",
    default="gloo",
    type=str,
    choices=["gloo", "mpi", "nccl"],)
parser.add_argument(
    "--init_method",
    default="tcp://localhost:1994",
    type=str)
parser.add_argument(
    "--port",
    default=None,
    type=int,
    help="If port is specified, the port in init_method will be replaced.")
parser.add_argument(
    "--world_size",
    default=2,
    type=int,
    help="If set with 2, the rank can be ignored.")
parser.add_argument(
    "--rank",
    "--local_rank",
    default=-1,
    type=int,
    help="If the world is 2, rank can be ignored.")
parser.add_argument(
    "--group_name",
    default="Admirable",
    type=str,
    help="Add a group name to better recognize each address.")
