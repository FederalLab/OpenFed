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
`openfed.tools.launch` is a module that spawns up multiple simulated
federated processes on each of the training nodes.

The utility can be used for single-node federated simulation, in which one or
more processes per node will be spawned. The utility can be used for either
CPU training or GPU training. If the utility is used for GPU training,
each federated process will be operating on a single GPU. This can achieve
well-improved single-node training performance. It can also be used in
multi-node federated training, by spawning up multiple processes on each node
for well-improved multi-node federated training performance as well.
This will especially be beneficial for systems with multiple Infiniband
interfaces that have direct-GPU support, since all of them can be utilized for
aggregated communication bandwidth.

In both cases of single-node federated training or multi-node federated
training, this utility will launch the given number of processes per node
(``--nproc_per_node``). If used for GPU training, this number needs to be less
or equal to the number of GPUs on the current system (``nproc_per_node``),
and each process will be operating on a single GPU from *GPU 0 to
GPU (nproc_per_node - 1)*.

**How to use this module:**

1. Single-Node multi-process federated training

::

    >>> python -m openfed.tools.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

2. Multi-Node multi-process federated training: (e.g. two nodes)


Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

::

    >>> python -m openfed.tools.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

Node 2:

::

    >>> python -m openfed.tools.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

3. To look up what optional arguments this module offers:

::

    >>> python -m openfed.tools.launch --help


**Important Notices:**

1. This utility and multi-process distributed (single-node or
multi-node) GPU training currently only achieves the best performance using
the NCCL distributed backend. Thus NCCL backend is the recommended backend to
use for GPU training.

2. In your training program, you must parse the command-line argument:
``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by this module.
If your training program uses GPUs, you should ensure that your code only
runs on the GPU device of LOCAL_PROCESS_RANK. This can be done by:

Parsing the local_rank argument

::

    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument("--local_rank", type=int)
    >>> args = parser.parse_args()

Set your device to local rank using either

::

    >>> torch.cuda.set_device(args.local_rank)  # before your code runs

or

::

    >>> with torch.cuda.device(args.local_rank):
    >>>    # your code to run

3. In your training program, you are supposed to call the following function
at the beginning to start the distributed backend. You need to make sure that
the init_method uses ``env://``, which is the only supported ``init_method``
by this module.

::

    openfed.core.space.Country.init_process_group(backend='YOUR BACKEND',
                                         init_method='env://')

4. Another way to pass ``local_rank`` to the subprocesses via environment variable
``LOCAL_RANK``. This behavior is enabled when you launch the script with
``--use_env=True``. You must adjust the subprocess example above to replace
``args.local_rank`` with ``os.environ['LOCAL_RANK']``; the launcher
will not pass ``--local_rank`` when you specify this flag.

.. warning::

    ``local_rank`` is NOT globally unique: it is only unique per process
    on a machine.  Thus, don't use it to decide if you should, e.g.,
    write to a networked filesystem.  See
    https://github.com/pytorch/pytorch/issues/12042 for an example of
    how things can go wrong if you don't do this correctly.

"""


import os
import signal
import subprocess
import sys
import time
from argparse import REMAINDER, ArgumentParser
from typing import IO, Any, List, Optional

from openfed.utils import openfed_title

node_local_rank_stdout_filename = "openfed_node_{}_local_rank_{}_stdout"
node_local_rank_stderr_filename = "openfed_node_{}_local_rank_{}_stderr"


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description=f"{openfed_title} federated simulation training launch "
                                        "helper utility that will spawn up "
                                        "multiple federated processes.")

    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for federated "
                             "training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node federated "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=1994, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communication during federated "
                             "training")
    parser.add_argument("--use_env", default=False, action="store_true",
                        help="Use environment variable to pass "
                             "'local rank'. For legacy reasons, the default value is False. "
                             "If set to True, the script will not pass "
                             "--local_rank as argument, and will instead set LOCAL_RANK.")
    parser.add_argument("-m", "--module", default=False, action="store_true",
                        help="Changes each process to interpret the launch script "
                             "as a python module, executing with the same behavior as"
                             "'python -m'.")
    parser.add_argument("--no_python", default=False, action="store_true",
                        help="Do not prepend the training script with \"python\" - just exec "
                             "it directly. Useful when the script is not a Python script.")
    parser.add_argument(
        "--logdir",
        default=None,
        type=str,
        help=f"""Relative path to write subprocess logs to. Passing in a relative
        path will create a directory if needed, and write the stdout and stderr to files
        {node_local_rank_stdout_filename} and {node_local_rank_stderr_filename}. Note that
        successive runs with the  same path to write logs to will overwrite existing logs,
        so be sure to save logs as needed. (The logs of rank 0 will be directly printed to the screen.)""",
    )

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # world size in terms of number of processes
    fed_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["FED_WORLD_SIZE"] = str(fed_world_size)

    processes: List[Any] = []

    if 'OMP_NUM_THREADS' not in os.environ and args.nproc_per_node > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print("*****************************************\n"
              "Setting OMP_NUM_THREADS environment variable for each process "
              "to be {} in default, to avoid your system being overloaded, "
              "please further tune the variable for optimal performance in "
              "your application as needed. \n"
              "*****************************************".format(current_env["OMP_NUM_THREADS"]))

    if args.logdir:
        # Possibly create the directory to write subprocess log output to.
        if os.path.exists(args.logdir):
            if not os.path.isdir(args.logdir):
                raise ValueError(
                    "argument --logdir must be a path to a directory.")
        else:
            # create the relative directory
            os.mkdir(os.path.join(os.getcwd(), args.logdir))

    subprocess_file_handles = []

    for local_rank in range(args.nproc_per_node):
        # each process's rank
        fed_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["FED_RANK"] = str(fed_rank)
        current_env["FED_LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        with_python = not args.no_python
        cmd = []
        if with_python:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
        else:
            if not args.use_env:
                raise ValueError(
                    "When using the '--no_python' flag, you must also set the '--use_env' flag.")
            if args.module:
                raise ValueError(
                    "Don't use both the '--no_python' flag and the '--module' flag at the same time.")

        cmd.append(args.training_script)

        if not args.use_env:
            cmd.append("--fed_local_rank={}".format(local_rank))
            cmd.append("--fed_world_size={}".format(fed_world_size))

        cmd.extend(args.training_script_args)

        stdout_handle: Optional[IO]
        stderr_handle: Optional[IO]
        if args.logdir:
            if fed_rank == 0:
                subprocess_file_handles.append((None, None))
            else:
                directory_path = os.path.join(os.getcwd(), args.logdir)
                node_rank = args.node_rank
                stdout_file_name = node_local_rank_stdout_filename.format(
                    node_rank, local_rank)
                stderr_file_name = node_local_rank_stderr_filename.format(
                    node_rank, local_rank)
                stdout_handle = open(os.path.join(
                    directory_path, stdout_file_name), "w")
                stderr_handle = open(os.path.join(
                    directory_path, stderr_file_name), "w")
                subprocess_file_handles.append((stdout_handle, stderr_handle))
                stdout_name = stdout_handle.name
                stderr_name = stderr_handle.name
                print(f"""Note: Stdout and stderr for node {node_rank} rank {local_rank} will
                be written to {stdout_name}, {stderr_name} respectively.""")

        sig_names = {2: "SIGINT", 15: "SIGTERM"}
        last_return_code = None

        def sigkill_handler(signum, frame):
            for process in processes:
                print(f"Killing subprocess {process.pid}")
                try:
                    process.kill()
                except Exception:
                    pass
            if last_return_code is not None and last_return_code != signal.SIGTERM:
                raise subprocess.CalledProcessError(
                    returncode=last_return_code, cmd=cmd)
            if signum in sig_names:
                print(f"Main process received {sig_names[signum]}, exiting")
            sys.exit(1)

        # pass SIGINT/SIGTERM to children if the parent is being terminated
        signal.signal(signal.SIGINT, sigkill_handler)
        signal.signal(signal.SIGTERM, sigkill_handler)

        stdout_handle = None if not subprocess_file_handles else subprocess_file_handles[
            local_rank][0]
        stderr_handle = None if not subprocess_file_handles else subprocess_file_handles[
            local_rank][1]
        process = subprocess.Popen(
            cmd, env=current_env, stdout=stdout_handle, stderr=stderr_handle)
        processes.append(process)

    try:
        alive_processes = set(processes)
        while len(alive_processes):
            finished_processes = []
            for process in alive_processes:
                if process.poll() is None:
                    # the process is still running
                    continue
                else:
                    if process.returncode != 0:
                        last_return_code = process.returncode  # for sigkill_handler
                        # not coming back
                        sigkill_handler(signal.SIGTERM, None)
                    else:
                        # exited cleanly
                        finished_processes.append(process)
            alive_processes = set(alive_processes) - set(finished_processes)

            time.sleep(1)
    finally:
        # close open file descriptors
        for (stdout_handle, stderr_handle) in subprocess_file_handles:
            if stdout_handle is not None:
                stdout_handle.close()
            if stderr_handle is not None:
                stderr_handle.close()


if __name__ == "__main__":
    main()
