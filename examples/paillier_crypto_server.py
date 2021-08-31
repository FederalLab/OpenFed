import os
import sys
sys.path.insert(0, '/Users/densechen/code/OpenFed/')


from openfed.hooks.cypher.paillier_crypto import key_gen
import os
import torch

if not os.path.isfile('/tmp/public.key') or not os.path.isfile('/tmp/private.key'):
    public_key, private_key = key_gen()
    torch.save(public_key, '/tmp/public.key')
    torch.save(private_key, '/tmp/private.key')
    print("Save new key to /tmp/public.key and /tmp/private.key")
else:
    private_key = torch.load('/tmp/private.key')
    print("Load private key from /tmp/private.key")
    print(private_key)


import torch.nn as nn

network = nn.Linear(784, 10)
loss_fn = nn.CrossEntropyLoss()

from openfed.optim import PaillierOp, build_aggregator

agg_op = PaillierOp(network.parameters(), private_key)

aggregator = build_aggregator(agg_op)

print(aggregator)

import torch

from openfed.optim import build_fed_optim

optim = torch.optim.SGD(network.parameters(), lr=1.0)
fed_optim = build_fed_optim(optim)

print(fed_optim)

import openfed

server_node = openfed.topo.Node('server', openfed.default_tcp_address, mtt=5)
client = openfed.topo.Node('client', openfed.empty_address, mtt=5)

topology = openfed.topo.Topology()
topology.add_edge(client, server_node)

federated_group_props = topology.topology_analysis(server_node)[0]

print(federated_group_props)

from openfed import API
openfed_api = API(
    state_dict=network.state_dict(keep_vars=True),
    fed_optim=fed_optim,
    aggregator=aggregator)

print(openfed_api)


from openfed.hooks import Aggregate

with openfed_api: 
    aggregate = Aggregate(
        activated_parts=dict(train=2),
        max_version=5,
    )
    print(aggregate)


for r in range(5):
    openfed_api.step()

openfed_api.finish()

print("Finished")