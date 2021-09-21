## Topo

### Node

Each device is regarded as a `Node` with `nick_name` and `address`. The nick name is the identification for each node and needs to be unique. Any nodes could connect to others via the address. Only when two nodes have the same address and nick name, we will regard them as the some one.

For example, we can define two nodes:

```shell
>>> import openfed
>>> alpha = openfed.topo.Node('alpha node', openfed.default_tcp_address)
>>> beta = openfed.topo.Node('beta node', openfed.default_file_address)
>>> alpha
<OpenFed> Node
nick name: alpha node
<OpenFed> Address
+---------+---------------------+------------+------+
| backend |     init_method     | world_size | rank |
+---------+---------------------+------------+------+
|   gloo  | tcp://localhost:... |     2      |  -1  |
+---------+---------------------+------------+------+


>>> beta
<OpenFed> Node
nick name: beta node
<OpenFed> Address
+---------+---------------------+------------+------+
| backend |     init_method     | world_size | rank |
+---------+---------------------+------------+------+
|   gloo  | file:///tmp/open... |     2      |  -1  |
+---------+---------------------+------------+------+
```

### Edge

The relation between two nodes is determined via `Edge`. An `Edge` with two attributions:

- `start`: The start node, namely the collaborator nodes.
- `end`: The end node, namely the aggregator nodes.

If you want to build a connection between alpha(collaborator) and beta(aggregator), you may need a piece of code like:

```shell
>>> edge = openfed.topo.Edge(alpha, beta)
>>> edge
<OpenFed> Edge
|alpha node -> beta node.
```

In `OpenFed`, all the connection relationship should be represented as a `Topology`.

### Topology

In `OpenFed`, we use `Topology` to manage massive nodes and edges. Here, we try to build a very simple centralized topology between three nodes, alpha(aggregator), beta(collaborator), gamma(collaborator).

```shell
>>> import openfed
>>> # define node
>>> alpha = openfed.topo.Node('alpha node', openfed.default_tcp_address)
>>> # the address of collaborator can be ignored.
>>> beta = openfed.topo.Node('beta node', openfed.empty_address)
>>> gamma = openfed.topo.Node('gamma node', openfed.empty_address)
>>> # define an empty topology
>>> topology = openfed.topo.Topology()
>>> # add nodes to topology
>>> topology.add_node(alpha)
>>> topology.add_node(beta)
>>> topology.add_node(gamma)
>>> # add edge
>>> topology.add_edge(beta, alpha)
>>> topology.add_edge(gamma, alpha)
>>> topology
+------------+------------+-----------+------------+
|   CO\AG    | alpha node | beta node | gamma node |
+------------+------------+-----------+------------+
| alpha node |     .      |     .     |     .      |
| beta node  |     ^      |     .     |     .      |
| gamma node |     ^      |     .     |     .      |
+------------+------------+-----------+------------+
```

### FederatedGroup

We will analysis `Topology` and build a `FederatedGroup` for each node. Whatever the topology is, we will divide it into many federated groups. In each group, the node can only be a `aggregator` or a `collaborator`. In different groups, the node can play different roles.

Federated groups of alpha node:

```shell
>>> federated_groups = openfed.topo.analysis(topology, alpha)
>>> federated_groups
[<OpenFed> FederatedProperties
+--------------------+------------+
|        role        | nick_name  |
+--------------------+------------+
| openfed_aggregator | alpha node |
+--------------------+------------+
<OpenFed> Address
+---------+---------------------+------------+------+
| backend |     init_method     | world_size | rank |
+---------+---------------------+------------+------+
|   gloo  | tcp://localhost:... |     3      |  0   |
+---------+---------------------+------------+------+

]
```

Federated groups of beta node:

```shell
>>> federated_groups = openfed.topo.analysis(topology, beta)
>>> federated_groups
[<OpenFed> FederatedProperties
+----------------------+-----------+
|         role         | nick_name |
+----------------------+-----------+
| openfed_collaborator | beta node |
+----------------------+-----------+
<OpenFed> Address
+---------+---------------------+------------+------+
| backend |     init_method     | world_size | rank |
+---------+---------------------+------------+------+
|   gloo  | tcp://localhost:... |     3      |  1   |
+---------+---------------------+------------+------+

]
```

Federated groups of gamma node:

```shell
>>> federated_groups = openfed.topo.analysis(topology, gamma)
>>> federated_groups
[<OpenFed> FederatedProperties
+----------------------+------------+
|         role         | nick_name  |
+----------------------+------------+
| openfed_collaborator | gamma node |
+----------------------+------------+
<OpenFed> Address
+---------+---------------------+------------+------+
| backend |     init_method     | world_size | rank |
+---------+---------------------+------------+------+
|   gloo  | tcp://localhost:... |     3      |  2   |
+---------+---------------------+------------+------+

]
```

You can refer to `openfed.tools.topo_builder` for more details about how to build a complex topology.
