# Copyright (c) FederalLab. All rights reserved.
r'''
`openfed.tools.simulator` is a module that spawns up multiple simulated
federated processes on each of the training nodes.
'''

import os
import signal
import subprocess
import sys
import time
from argparse import REMAINDER, ArgumentParser
from typing import IO, Any, List, Optional

from openfed.topo import Topology
from openfed.utils import openfed_title

node_local_rank_stdout_filename = 'openfed_node_{}_local_rank_{}_stdout'
node_local_rank_stderr_filename = 'openfed_node_{}_local_rank_{}_stderr'


def parse_args():
    """Helper function parsing the command line options.

    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description=f'{openfed_title} federated simulation training launch '
        'helper utility that will spawn up '
        'multiple federated processes.')

    # Optional arguments for the launch helper
    parser.add_argument(
        '--topology',
        type=str,
        required=True,
        help='The topology file to use for training.')
    parser.add_argument(
        '-m',
        '--module',
        default=False,
        action='store_true',
        help='Changes each process to interpret the launch script '
        'as a python module, executing with the same behavior as'
        '`python -m`.')
    parser.add_argument(
        '--no_python',
        default=False,
        action='store_true',
        help='Do not prepend the training script with \'python\' - just exec '
        'it directly. Useful when the script is not a Python script.')
    parser.add_argument(
        '--logdir',
        default=None,
        type=str,
        help=f'''Relative path to write subprocess logs to.
        Passing in a relative path will create a directory if needed,
        and write the stdout and stderr to files
        {node_local_rank_stdout_filename} and {node_local_rank_stderr_filename}
        Note that successive runs with the  same path to write logs to will
        overwrite existing logs, so be sure to save logs as needed.
        (The logs of rank 0 will be directly printed to the screen.)''',
    )

    # positional
    parser.add_argument(
        'training_script',
        type=str,
        help='The full path to the single GPU training '
        'program/script to be launched in parallel, '
        'followed by all the arguments for the '
        'training script')

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    topology = Topology().load(args.topology)

    processes: List[Any] = []

    if args.logdir:
        # Possibly create the directory to write subprocess log output to.
        if os.path.exists(args.logdir):
            if not os.path.isdir(args.logdir):
                raise ValueError(
                    'argument --logdir must be a path to a directory.')
        else:
            # create the relative directory
            os.mkdir(os.path.join(os.getcwd(), args.logdir))

    subprocess_file_handles = []

    def sigkill_handler(signum, *args):
        for process in processes:
            print(f'Killing subprocess {process.pid}')
            try:
                process.kill()
            except Exception:
                pass
        if last_return_code is not None and last_return_code != signal.SIGTERM:
            raise subprocess.CalledProcessError(
                returncode=last_return_code, cmd=cmd)
        if signum in sig_names:
            print(f'Main process received {sig_names[signum]}, exiting')
        sys.exit(1)

    for local_rank in range(len(topology.nodes)):
        # spawn the processes
        with_python = not args.no_python
        cmd = []
        if with_python:
            cmd = [sys.executable, '-u']
            if args.module:
                cmd.append('-m')
        else:
            if not args.use_env:
                raise ValueError('When using the `--no_python` flag, '
                                 'you must also set the `--use_env` flag.')
            if args.module:
                raise ValueError('Do not use both the `--no_python` flag '
                                 'and the `--module` flag at the same time.')

        cmd.append(args.training_script)

        cmd.extend(args.training_script_args)
        cmd.append(f'--nick_name={topology.nodes[local_rank].nick_name}')
        cmd.append(f'--topology={args.topology}')

        stdout_handle: Optional[IO]
        stderr_handle: Optional[IO]
        if args.logdir:
            if local_rank == 0:
                subprocess_file_handles.append((None, None))
            else:
                directory_path = os.path.join(os.getcwd(), args.logdir)
                node_rank = args.node_rank
                stdout_file_name = node_local_rank_stdout_filename.format(
                    node_rank, local_rank)
                stderr_file_name = node_local_rank_stderr_filename.format(
                    node_rank, local_rank)
                stdout_handle = open(
                    os.path.join(directory_path, stdout_file_name), 'w')
                stderr_handle = open(
                    os.path.join(directory_path, stderr_file_name), 'w')
                subprocess_file_handles.append((stdout_handle, stderr_handle))
                stdout_name = stdout_handle.name
                stderr_name = stderr_handle.name
                print(
                    'Note: Stdout and stderr for '
                    f'node {node_rank} rank {local_rank} will '
                    f'be written to {stdout_name}, {stderr_name} respectively.'
                )

        sig_names = {2: 'SIGINT', 15: 'SIGTERM'}
        last_return_code = None

        # pass SIGINT/SIGTERM to children if the parent is being terminated
        signal.signal(signal.SIGINT, sigkill_handler)
        signal.signal(signal.SIGTERM, sigkill_handler)

        stdout_handle = None if not subprocess_file_handles\
            else subprocess_file_handles[local_rank][0]
        stderr_handle = None if not subprocess_file_handles\
            else subprocess_file_handles[local_rank][1]
        process = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            stdout=stdout_handle,
            stderr=stderr_handle)
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
                        # for sigkill_handler
                        last_return_code = process.returncode
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


if __name__ == '__main__':
    main()
