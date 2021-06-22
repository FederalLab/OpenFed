# OpenFed

## Bugs

1. It seems that when communicate with progress in different GPU devices, upload and download operations will get stuck.

## Install

Python>=3.7, PyTorch=1.9.0 are required.

```
conda create -n openfed python=3.7 -y
conda activate openfed
pip3 install -r requirements.txt

# test
python3 -m openfed.launch --nproc_per_node 3 --logdir /tmp --server_output demo.py
```

## Say hello to OpenFed

```bash
python -m openfed.launch --nproc_per_node 11 --logdir /tmp --server_output demo.py

*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
Note: Stdout and stderr for node 0 rank 1 will
                be written to /tmp/node_0_local_rank_1_stdout, /tmp/node_0_local_rank_1_stderr respectively.
...

World: 1
Country: 0
Maintainer: 1
Thread: 2

2021-06-21 19:10:50.867 | INFO     | openfed.federated.joint:safe_run:49 - Waiting
<OpenFed> Address
@ Admirable

[W ProcessGroupGloo.cpp:559] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
2021-06-21 19:10:51.171 | INFO     | openfed.federated.inform.informer:collect:202 - 
2021-06-21 19:10:51.195 | INFO     | openfed.federated.inform.informer:collect:202 - System Information List
System: Darwin
Platform: Darwin
Version: Darwin Kernel Version 20.5.0: Sat May  8 05:10:33 PDT 2021; root:xnu-7195.121.3~9/RELEASE_X86_64
Architecture: ['64bit', '']
Machine: x86_64
Node: C02DW0CQMD6R
Processor: i386

...

2021-06-21 19:10:51.315 | INFO     | openfed.federated.joint:safe_run:89 - Connected
<OpenFed> Address
+---------+----------------------+------------+------+-------+------------+
| Backend |     Init Method      | World Size | Rank | Store | Group Name |
+---------+----------------------+------------+------+-------+------------+
|   gloo  | tcp://localhost:1994 |     11     |  0   |  None | Admirable  |
+---------+----------------------+------------+------+-------+------------+

2021-06-21 19:10:51.823 | INFO     | openfed.backend:step_after_download:173 - Recieve Model
@1
From <OpenFed> Reign
Version: 0
Status: PULL

...

2021-06-21 19:10:57.891 | INFO     | openfed.common.unify:finish:72 - Finished.
<OpenFed> OpenFed Unified API
<OpenFed> Maintainer
0 in pending
1 in finished
0 in discard
```

## Project Structure

```bash
openfed
├── __init__.py
├── aggregate
│   ├── __init__.py
│   ├── aggregator.py
│   ├── average.py
│   ├── elastic.py
│   └── naive.py
├── api.py
├── backend.py
├── common
│   ├── __init__.py
│   ├── address.py
│   ├── array.py
│   ├── constants.py
│   ├── exception.py
│   ├── hook.py
│   ├── logging.py
│   ├── package.py
│   ├── parser.py
│   ├── peeper.py
│   ├── thread.py
│   ├── unify.py
│   ├── vars.py
│   └── wrapper.py
├── data
│   ├── __init__.py
│   ├── dataset.py
│   ├── nlp
│   │   ├── __init__.py
│   │   ├── shakespear.py
│   │   └── stackoverflow.py
│   ├── partitioner.py
│   └── vision
│       ├── __init__.py
│       └── emnist.py
├── federated
│   ├── __init__.py
│   ├── country.py
│   ├── deliver
│   │   ├── __init__.py
│   │   ├── delivery.py
│   │   └── functional.py
│   ├── destroy.py
│   ├── functional.py
│   ├── inform
│   │   ├── __init__.py
│   │   ├── functional.py
│   │   └── informer.py
│   ├── joint.py
│   ├── lock.py
│   ├── maintainer.py
│   ├── register.py
│   ├── reign.py
│   └── world.py
├── frontend.py
├── helper.py
├── launch.py
├── optim
│   ├── __init__.py
│   └── elastic_aux.py
└── utils
    ├── __init__.py
    └── utils.py

10 directories, 53 files
```