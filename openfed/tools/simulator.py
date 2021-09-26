# @Author            : FederalLab
# @Date              : 2021-09-25 16:54:04
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:54:04
# Copyright (c) FederalLab. All rights reserved.
r'''
`openfed.tools.simulator` is a module that spawns up multiple simulated
federated processes on each of the training nodes.
'''

import json
import os
import signal
import subprocess
import sys
import time
from argparse import REMAINDER, ArgumentParser
from typing import IO, Any, List, Optional

import openfed
import openfed.topo as topo
from openfed.utils import FMT

node_stdout_filename = 'openfed_node_{}_stdout'
node_stderr_filename = 'openfed_node_{}_stderr'


def parse_args():
    """Helper function parsing the command line options."""
    parser = ArgumentParser(
        description=f'{FMT.openfed_title} federated simulation training launch'
        'helper utility that will spawn up '
        'multiple federated processes.')

    parser.add_argument(
        '--nproc',
        type=int,
        default=2,
        help='The number of processes to launch on each node, '
        'for GPU training, this is recommended to be set '
        'to the number of GPUs in your system so that '
        'each process can be bound to a single GPU.')

    # Optional arguments for the launch helper
    parser.add_argument(
        '--logdir',
        default=None,
        type=str,
        help=f'''Relative path to write subprocess logs to.
        Passing in a relative path will create a directory if needed,
        and write the stdout and stderr to files
        {node_stdout_filename} and {node_stderr_filename}
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


def build_centralized_topology(nproc):
    assert nproc >= 2, 'nproc must be greater than 2'

    # build node
    aggregator = topo.Node('aggregator', openfed.default_file_address)
    collaborators = [
        topo.Node(f'collaborator-{i}', openfed.empty_address)
        for i in range(1, nproc)
    ]

    # build topo
    topology = topo.Topology(collaborators)
    topology.add_node(aggregator)

    # build edge
    for collaborator in collaborators:
        topology.add_edge(collaborator, aggregator)

    aggregator_props = topo.analysis(topology, aggregator)

    with open('/tmp/aggregator.json', 'w') as f:
        json.dump([props.serialize() for props in aggregator_props], f)

    for collaborator in collaborators:
        collaborator_props = topo.analysis(topology, collaborator)

        with open(f'/tmp/{collaborator.nick_name}.json', 'w') as f:
            json.dump([props.serialize() for props in collaborator_props], f)


def main():
    args = parse_args()
    if os.path.isfile('/tmp/openfed.sharedfile'):
        os.remove('/tmp/openfed.sharedfile')
    build_centralized_topology(args.nproc)

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

    for rank in range(args.nproc):
        node_name = 'aggregator' if rank == 0 else f'collaborator-{rank}'
        # spawn the processes
        cmd = [sys.executable, '-u']
        cmd.append(args.training_script)

        cmd.extend(args.training_script_args)
        cmd.append('--props={}'.format(f'/tmp/{node_name}.json'))

        stdout_handle: Optional[IO]
        stderr_handle: Optional[IO]
        if args.logdir:
            if rank == 0:
                subprocess_file_handles.append((None, None))
            else:
                directory_path = os.path.join(os.getcwd(), args.logdir)
                stdout_file_name = node_stdout_filename.format(node_name)
                stderr_file_name = node_stderr_filename.format(node_name)
                stdout_handle = open(
                    os.path.join(directory_path, stdout_file_name), 'w')
                stderr_handle = open(
                    os.path.join(directory_path, stderr_file_name), 'w')
                subprocess_file_handles.append((stdout_handle, stderr_handle))
                stdout_name = stdout_handle.name
                stderr_name = stderr_handle.name
                print(
                    f'Note: Stdout and stderr for {node_name} will '
                    f'be written to {stdout_name}, {stderr_name} respectively.'
                )

        sig_names = {2: 'SIGINT', 15: 'SIGTERM'}
        last_return_code = None

        # pass SIGINT/SIGTERM to children if the parent is being terminated
        signal.signal(signal.SIGINT, sigkill_handler)
        signal.signal(signal.SIGTERM, sigkill_handler)

        stdout_handle = None if not subprocess_file_handles\
            else subprocess_file_handles[rank][0]
        stderr_handle = None if not subprocess_file_handles\
            else subprocess_file_handles[rank][1]
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
