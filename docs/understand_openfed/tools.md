# Tools

## TopoBuilder

`TopoBuilder` provides a common line for you to build a massive topology graph more easily.
Then you can save it to disk and load it in your code.

The following example shows how to build a hierarchical topology graph:

<div align=center>
<img src="https://github.com/FederalLab/OpenFed/raw/main/docs/_static/image/topology.png" width="100" />
</div>

```shell
(openfed) python -m openfed.tools.topo_builder
A script to build topology.
<OpenFed>: add_node
Nick Name
red
Does this node requires address? (Y/n)
n
<OpenFed> Node
nick name: red
<OpenFed> Address
+---------+-------------+------------+------+
| backend | init_method | world_size | rank |
+---------+-------------+------------+------+
|   null  |     null    |     2      |  -1  |
+---------+-------------+------------+------+


<OpenFed>: add_node
Nick Name
green
Does this node requires address? (Y/n)
n
<OpenFed> Node
nick name: green
<OpenFed> Address
+---------+-------------+------------+------+
| backend | init_method | world_size | rank |
+---------+-------------+------------+------+
|   null  |     null    |     2      |  -1  |
+---------+-------------+------------+------+


<OpenFed>: add_node
Nick Name
purple
Does this node requires address? (Y/n)
n
<OpenFed> Node
nick name: purple
<OpenFed> Address
+---------+-------------+------------+------+
| backend | init_method | world_size | rank |
+---------+-------------+------------+------+
|   null  |     null    |     2      |  -1  |
+---------+-------------+------------+------+


<OpenFed>: add_node
Nick Name
blue
Does this node requires address? (Y/n)
y
Backend (gloo, mpi, nccl)
gloo
Init method i.e., tcp://localhost:1994, file:///tmp/openfed.sharedfile)
tcp://localhost:1994
<OpenFed> Node
nick name: blue
<OpenFed> Address
+---------+---------------------+------------+------+
| backend |     init_method     | world_size | rank |
+---------+---------------------+------------+------+
|   gloo  | tcp://lo...ost:1994 |     2      |  -1  |
+---------+---------------------+------------+------+


<OpenFed>: add_node
Nick Name
yellow
Does this node requires address? (Y/n)
y
Backend (gloo, mpi, nccl)
mpi
Init method i.e., tcp://localhost:1994, file:///tmp/openfed.sharedfile)
file://tmp/openfed.sharedfile
<OpenFed> Node
nick name: yellow
<OpenFed> Address
+---------+---------------------+------------+------+
| backend |     init_method     | world_size | rank |
+---------+---------------------+------------+------+
|   mpi   | file://t...aredfile |     2      |  -1  |
+---------+---------------------+------------+------+


<OpenFed>: build_edge
Start node nick name
red
End node nick name
blue
<OpenFed> Edge
|red -> blue.

<OpenFed>: build_edge
Start node nick name
green
End node nick name
blue
<OpenFed> Edge
|green -> blue.

<OpenFed>: build_edge
Start node nick name
blue
End node nick name
yellow
<OpenFed> Edge
|blue -> yellow.

<OpenFed>: build_edge
Start node nick name
purple
End node nick name
yellow
<OpenFed> Edge
|purple -> yellow.

<OpenFed>: save
Filename:
topology
+--------+-----+-------+--------+------+--------+
| CO\AG  | red | green | purple | blue | yellow |
+--------+-----+-------+--------+------+--------+
|  red   |  .  |   .   |   .    |  ^   |   .    |
| green  |  .  |   .   |   .    |  ^   |   .    |
| purple |  .  |   .   |   .    |  .   |   ^    |
|  blue  |  .  |   .   |   .    |  .   |   ^    |
| yellow |  .  |   .   |   .    |  .   |   .    |
+--------+-----+-------+--------+------+--------+
<OpenFed>: analysis
Folder to save the analysis result:
props
Processing red
[{'role': 'openfed_collaborator', 'nick_name': 'red', 'address': {'backend': 'gloo', 'init_method': 'tcp://localhost:1994', 'world_size': 3, 'rank': 2}}]
Processing green
[{'role': 'openfed_collaborator', 'nick_name': 'green', 'address': {'backend': 'gloo', 'init_method': 'tcp://localhost:1994', 'world_size': 3, 'rank': 1}}]
Processing purple
[{'role': 'openfed_collaborator', 'nick_name': 'purple', 'address': {'backend': 'mpi', 'init_method': 'file://tmp/openfed.sharedfile', 'world_size': 3, 'rank': 2}}]
Processing blue
[{'role': 'openfed_aggregator', 'nick_name': 'blue', 'address': {'backend': 'gloo', 'init_method': 'tcp://localhost:1994', 'world_size': 3, 'rank': 0}}, {'role': 'openfed_collaborator', 'nick_name': 'blue', 'address': {'backend': 'mpi', 'init_method': 'file://tmp/openfed.sharedfile', 'world_size': 3, 'rank': 1}}]
Processing yellow
[{'role': 'openfed_aggregator', 'nick_name': 'yellow', 'address': {'backend': 'mpi', 'init_method': 'file://tmp/openfed.sharedfile', 'world_size': 3, 'rank': 0}}]
<OpenFed>: exit
```

## Simulator

`Simulator`, which is similar with `torch.distributed.launch`, is a module that spawns up multiple federated
training processes on each of the training nodes.
It will build a centralized topology automatically. It is very useful while simulating massive nodes to do the federated learning experience.

Write a piece of code, named `run.py`:

```python
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--props')

args = parser.parse_args()

print(args.props)
```

Usage:

```shell
(openfed) python -m openfed.tools.simulator --nproc 10 run.py
/tmp/aggregator.json
/tmp/collaborator-1.json
/tmp/collaborator-2.json
/tmp/collaborator-3.json
/tmp/collaborator-4.json
/tmp/collaborator-5.json
/tmp/collaborator-6.json
/tmp/collaborator-7.json
/tmp/collaborator-8.json
/tmp/collaborator-9.json
```
