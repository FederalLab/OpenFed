# Federated

## Pipe

:class:`Pipe` maintains the communication operation between two nodes, including tensor data and info message.
It uses a store to transfer info message and process group with `gloo` or `mpi` to transfer tensor data.

## DistributedProperties

:class:`DistributedProperties` contains all distributed attributions of `torch.distributed.distributed_c10d`.
Usually, you can use it with context environment.

```python
with dist_props:
    ...
```

## FederatedProperties

:class:`FederatedProperties` contains all federated attributions, such as address, role and nick name.
It is usually generated via :func:`openfed.topo.analysis`.

## Examples

Here, we try to communicate some information among `aggregator`, `collaborator_alpha` and `collaborator_beta`.
You need to open three independent terminals to run the following three scripts.

Aggregator:

```python
# build a topology first
import time

# transfer data
import torch

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

# build pipe
pipes = openfed.federated.init_federated_group(federated_group_prop)

assert len(pipes) == 2
alpha_pipe, beta_pipe = pipes

# transfer message
alpha_pipe.direct_set('message_0', 'hello world from aggregator to alpha')
beta_pipe.direct_set('message_0', 'hello world from aggregator to beta')

print(alpha_pipe.direct_get('message_1'))
print(beta_pipe.direct_get('message_1'))

data = torch.tensor(-1)
with alpha_pipe.dist_props:
    time.sleep(0.5)
    # send data to alpha
    alpha_pipe.upload(data)

    time.sleep(0.5)
    # download data from alpha
    assert alpha_pipe.download() == 1

with beta_pipe.dist_props:
    time.sleep(0.5)
    # send data to beta
    beta_pipe.upload(data)

    time.sleep(0.5)
    # download data from beta
    assert beta_pipe.download() == 2

time.sleep(1)
```

Collaborator alpha:

```python
# build a topology first
import time

# transfer tensor
import torch

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

# build pipe
pipes = openfed.federated.init_federated_group(federated_group_prop)

alpha_pipe = pipes[0]

# transfer message
print(alpha_pipe.direct_get('message_0'))

alpha_pipe.direct_set('message_1', 'hello world from alpha to aggregator')

data = torch.tensor(1)
with alpha_pipe.dist_props:
    # download data from aggregator
    assert alpha_pipe.download() == -1

    # upload data to aggregator
    alpha_pipe.upload(data)

time.sleep(1)
```

Collaborator beta:

```python
# build a topology first
import time

# transfer data
import torch

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

# build pipe
pipes = openfed.federated.init_federated_group(federated_group_prop)

beta_pipe = pipes[0]

# transfer message
print(beta_pipe.direct_get('message_0'))

beta_pipe.direct_set('message_1', 'hello world from beta to aggregator')

data = torch.tensor(2)
with beta_pipe.dist_props:
    # download data from aggregator
    assert beta_pipe.download() == -1

    # upload data to aggregator
    beta_pipe.upload(data)

time.sleep(1)
```

The output of aggregator:

```shell
(openfed)  python aggregator.py
hello world from alpha to aggregator
hello world from beta to aggregator
```

The output of collaborator alpha:

```shell
(openfed) python collaborator_alpha.py
hello world from aggregator to alpha
```

The output of collaborator beta:

```shell
(openfed) python collaborator_beta.py
hello world from aggregator to beta
```
