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


from openfed.core import World, leader

world = World(role=leader, dal=False, mtt=5)

print(world)


from openfed import API
openfed_api = API(
    world=world,
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

from openfed.common import default_tcp_address

address = default_tcp_address

print(address)

import time
openfed_api.build_connection(address=address)

print(openfed_api.federated_group)


openfed_api.run()

openfed_api.finish()

print("Finished")