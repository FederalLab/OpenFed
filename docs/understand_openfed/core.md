# Core

## Maintainer

:class:`Maintainer` bridges the connection between upper(federated algorithms) and lower(communication and topology) layers. It has the following properties:

- `pipe`: The currently target to communicate with. A maintainer will manage several pipes in the same time, and `pipe` will indicate what is the current target.
- `pipes`: A list of pipes to communicate with.
- `current_step`: It is used to indicate which step is running on.
- `fed_props`: Actually, a maintainer is corresponding to a specified federated group. We record the related federated group properties in this attributions.

You can use :class:`Maintainer` to conduct a flexible communication with other nodes more easily than :class:`Pipe`.

## Examples

Aggregator:

```python
    # build a topology first
    import openfed
    import openfed.topo as topo

    aggregator = topo.Node('aggregator', openfed.default_tcp_address)
    alpha = topo.Node('alpha', openfed.empty_address)
    beta = topo.Node('beta', openfed.empty_address)

    topology = topo.Topology()
    topology.add_node_list([aggregator, alpha, beta])
    topology.add_edge(alpha, aggregator)
    topology.add_edge(beta, aggregator)

    # analysis topology to get federated group props
    federated_group_props = topo.analysis(topology, aggregator)
    assert len(federated_group_props) == 1
    federated_group_prop = federated_group_props[0]

    # build network
    import torch.nn as nn
    network = nn.Linear(10, 1)

    # build maintainer
    from openfed.core import Maintainer
    maintainer = Maintainer(federated_group_prop,
                            network.state_dict(keep_vars=True))

    with maintainer:
        openfed.functional.device_alignment()
        openfed.functional.count_step(2)

    maintainer.step()
```

Collaborator alpha:

```python
    # build a topology first
    import openfed
    import openfed.topo as topo

    aggregator = topo.Node('aggregator', openfed.default_tcp_address)
    alpha = topo.Node('alpha', openfed.empty_address)
    beta = topo.Node('beta', openfed.empty_address)

    topology = topo.Topology()
    topology.add_node_list([aggregator, alpha, beta])
    topology.add_edge(alpha, aggregator)
    topology.add_edge(beta, aggregator)

    # analysis topology to get federated group props
    federated_group_props = topo.analysis(topology, alpha)
    assert len(federated_group_props) == 1
    federated_group_prop = federated_group_props[0]

    # build network
    import torch.nn as nn
    network = nn.Linear(10, 1)

    # build maintainer
    from openfed.core import Maintainer
    maintainer = Maintainer(federated_group_prop,
                            network.state_dict(keep_vars=True))

    with maintainer:
        openfed.functional.device_alignment()

    maintainer.step(upload=False)
    maintainer.package()
    maintainer.step(download=False)
```

Collaborator beta:

```python
    # build a topology first
    import openfed
    import openfed.topo as topo

    aggregator = topo.Node('aggregator', openfed.default_tcp_address)
    alpha = topo.Node('alpha', openfed.empty_address)
    beta = topo.Node('beta', openfed.empty_address)

    topology = topo.Topology()
    topology.add_node_list([aggregator, alpha, beta])
    topology.add_edge(alpha, aggregator)
    topology.add_edge(beta, aggregator)

    # analysis topology to get federated group props
    federated_group_props = topo.analysis(topology, beta)
    assert len(federated_group_props) == 1
    federated_group_prop = federated_group_props[0]

    # build network
    import torch.nn as nn
    network = nn.Linear(10, 1)

    # build maintainer
    from openfed.core import Maintainer
    maintainer = Maintainer(federated_group_prop,
                            network.state_dict(keep_vars=True))

    with maintainer:
        openfed.functional.device_alignment()

    maintainer.step(upload=False)
    maintainer.package()
    maintainer.step(download=False)
```
