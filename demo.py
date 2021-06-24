import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

import openfed
import openfed.aggregate as aggregate

parser = openfed.parser
args = parser.parse_args()

# specify a api type.
openfed_api = openfed.API(frontend=args.rank > 0)

# build the connection
openfed_api.build_connection(address=openfed.Address(args=args))

# build a network
net = nn.Linear(1, 1)
# net.cuda()

# define optimizer (use same optimizer in both server and client)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# define aggregator (actually, this is only used for server end)
aggregator = aggregate.AverageAggregator(
    net.parameters(), other_keys="momentum_buffer")

# tell openfed_api which part of data is needed to be cared.
openfed_api.set_state_dict(net.state_dict(keep_vars=True))

# set optimizer and aggregator for backend.
# only backend will run this code, frontend will skip.
openfed_api.set_optimizer(optimizer)
openfed_api.set_aggregator(aggregator)

# if this process is backend, it will go into this function
# and occupy this process. after finished, it will automatically exit.
# if you want to make it as thread to run this, just call openfed_api.start()
with openfed_api:
    openfed_api.run()

    # do 100 simulation
    for i in range(1, random.randint(10, 70)):
        print(f"Train @{i}")
        # download a new model
        if not openfed_api.download():
            break

        # reset
        optimizer.zero_grad()

        # training
        net(torch.randn(128, 1, 1)).sum().backward()
        optimizer.step()

        # submit
        openfed_api.pack_state(optimizer, keys=['momentum_buffer'])

        openfed_api.set_task_info({"train_instances": random.randint(1, 200)})

        if not openfed_api.upload():
            break

# finished
openfed_api.finish()
print(f"Finished. Exit Client @{args.rank}.")
