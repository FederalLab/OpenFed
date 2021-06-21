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
    default=-1,
    type=int,
    help="If the world is 2, rank can be ignored.")
parser.add_argument(
    "--group_name",
    default="Admirable",
    type=str,
    help="Add a group name to better recognize each address.")