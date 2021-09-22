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
