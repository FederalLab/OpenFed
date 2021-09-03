import os
import sys

sys.path.insert(0, '/Users/densechen/code/OpenFed/')

import os

import torch
from openfed.functional import PrivateKey, key_gen

if not os.path.isfile('/tmp/public.key') or not os.path.isfile(
        '/tmp/private.key'):
    public_key, private_key = key_gen()
    public_key.save('/tmp/public.key')
    private_key.save('/tmp/private.key')
    print("Save new key to /tmp/public.key and /tmp/private.key")
else:
    private_key = PrivateKey.load('/tmp/private.key')
    print("Load private key from /tmp/private.key")
print(private_key)

import torch.nn as nn

network = nn.Linear(784, 10)
loss_fn = nn.CrossEntropyLoss()

import torch
from openfed.federated import leader
from openfed.optim import FederatedOptimizer

optim = torch.optim.SGD(network.parameters(), lr=1.0)
fed_optim = FederatedOptimizer(optim, role=leader)
print(fed_optim)

import openfed

server_node = openfed.topo.Node('server', openfed.default_tcp_address)
client = openfed.topo.Node('client', openfed.empty_address)

topology = openfed.topo.Topology()
topology.add_edge(client, server_node)

fed_props = openfed.topo.analysis(topology, server_node)[0]

print(fed_props)

from openfed.core import Maintainer

mt = Maintainer(fed_props, network.state_dict(keep_vars=True))

with mt:
    openfed.F.device_alignment()
    openfed.F.count_step(2)

print(mt)

openfed.api(mt,
            fed_optim,
            5,
            agg_func=openfed.F.paillier_aggregation,
            private_key=private_key)

print("Finished")
