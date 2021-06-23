# OpenFed

## Bugs

1. It seems that when communicate with progress in different GPU devices, upload and download operations will get stuck.

## Install

Python>=3.7, PyTorch=1.9.0 are required.

```bash
conda create -n openfed python=3.7 -y
conda activate openfed
pip3 install -r requirements.txt

# test
python3 -m openfed.launch --nproc_per_node 3 --logdir /tmp --server_output demo.py

# robust: Create 101 nodes at the same time.
# this is not allowed under tcp mode for long latency.
# NOTE: If you use sharefile system, before start your program, 
# you have to remove the old sharefile first!
python3 -m openfed.launch --nproc_per_node 11 --logdir /tmp --server_output demo.py --init_method file:///tmp/openfed.sharefile
```

## Run demo step by step

Run "Python: client-2/1" and "Python: server" provided in `.vscode/launch.json`.
**Remember to remove the old share file first.**

## Say hello to OpenFed

```bash
python -m openfed.launch --nproc_per_node 3 --logdir /tmp --server_output demo.py --init_method file:///tmp/openfed.sharefile  
...
+-------+---------+------------+--------+
| World | Country | Maintainer | Thread |
+-------+---------+------------+--------+
|   1   |    0    |     1      |   2    |
+-------+---------+------------+--------+
2021-06-22 23:38:38.533 | INFO     | openfed.federated.joint:safe_run:51 - Waiting
<OpenFed> Address
@ Admirable

2021-06-22 23:38:38.627 | INFO     | openfed.federated.inform.informer:collect:279 - <OpenFed> Collector.SystemInfo
+--------+----------+--------------------+---------------+
| System | Platform |      Version       |  Architecture |
+--------+----------+--------------------+---------------+
| Darwin |  Darwin  | Darwin Kernel V... | ['64bit', ''] |
+--------+----------+--------------------+---------------+
+---------+--------------+-----------+
| Machine |     Node     | Processor |
+---------+--------------+-----------+
|  x86_64 | C02DW0CQMD6R |    i386   |
+---------+--------------+-----------+

...

2021-06-22 23:38:38.851 | INFO     | openfed.federated.joint:safe_run:121 - Connected
<OpenFed> Address
+---------+--------------------+------------+------+-------+------------+
| Backend |    Init Method     | World Size | Rank | Store | Group Name |
+---------+--------------------+------------+------+-------+------------+
|   gloo  | file:///tmp/ope... |     3      |  0   |  None | Admirable  |
+---------+--------------------+------------+------+-------+------------+

2021-06-22 23:38:39.161 | INFO     | openfed.backend:step_after_download:173 - Receive Model
@1
From <OpenFed> Reign
Version: 0
Status: PULL

...

2021-06-22 23:39:02.359 | INFO     | openfed.backend:step_after_download:173 - Receive Model
@100
From <OpenFed> Reign
Version: 0
Status: PULL

2021-06-22 23:39:02.475 | INFO     | openfed.common.unify:finish:79 - Finished.
<OpenFed> OpenFed Unified API
<OpenFed> Maintainer
+---------+----------+---------+
| Pending | Finished | Discard |
+---------+----------+---------+
|    0    |    1     |    0    |
+---------+----------+---------+

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
│   ├── vars.py
│   └── wrapper.py
├── data
│   ├── __init__.py
│   ├── dataset.py
│   ├── nlp
│   │   ├── __init__.py
│   │   ├── shakespeare.py
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
├── helper.py
├── launch.py
├── optim
│   ├── __init__.py
│   └── elastic_aux.py
├── unified
│   ├── __init__.py
│   ├── backend.py
│   ├── frontend.py
│   ├── step_hooks
│   │   ├── __init__.py
│   │   ├── step_after_destroy.py
│   │   ├── step_after_download.py
│   │   ├── step_after_upload.py
│   │   ├── step_at_failed.py
│   │   ├── step_at_last.py
│   │   ├── step_at_new_episode.py
│   │   ├── step_at_invalid_state.py
│   │   ├── step_at_zombie.py
│   │   ├── step_before_destroy.py
│   │   ├── step_before_download.py
│   │   └── step_before_upload.py
│   └── unify.py
└── utils
    ├── __init__.py
    ├── keyboard.py
    ├── table.py
    └── utils.py

12 directories, 68 files
```