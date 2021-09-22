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
