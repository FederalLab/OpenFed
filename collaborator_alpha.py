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
