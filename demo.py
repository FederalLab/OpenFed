import random

import torch
import torch.nn as nn
import torch.optim as optim

# >>> Import OpenFed
import openfed
import openfed.aggregate as aggregate
from openfed.utils import time_string

# >>> Get default arguments from OpenFed
args = openfed.parser.parse_args()

# >>> Specify an API for building federated learning
openfed_api = openfed.API(frontend=args.rank > 0)

# >>> Connect to Address.
openfed_api.build_connection(address=openfed.Address(args=args))

# Build Network
net = nn.Linear(1, 1)

# Define optimizer (use the same optimizer in both server and client)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# Define aggregator (actually, this is only used for server end)
aggregator = aggregate.AverageAggregator(net.parameters())

# >>> Set optimizer and aggregator for federated learning.
openfed_api.set_optimizer(optimizer)
openfed_api.set_aggregator(aggregator)

# >>> Tell OpenFed API which data should be transferred.
openfed_api.set_state_dict(net.state_dict(keep_vars=True))

# Context `with openfed_api` will go into the specified settings about openfed_api.
# Otherwise, will use the default one which shared by global OpenFed world.
with openfed_api:

    # >>> If openfed_api is a backend, call `run()` will go into the loop ring.
    # >>> Call `start()` will run it as a thread.
    # >>> If openfed_api is a frontend, call `run()` will directly skip this function automatically.
    openfed_api.run()

    # Do simulation random times at [10, 70].
    for i in range(1, random.randint(10, 70)):
        print(f"{time_string()}: Simulation @{i}")

        # Download latest model.
        print(f"{time_string()}: Downloading latest model from server.")
        if not openfed_api.download():
            print(f"Downloading failed.")
            break
        
        # Downloaded
        print(f"{time_string()}: Downloaded!")

        # Start a standard forward/backward pass.
        optimizer.zero_grad()
        net(torch.randn(128, 1, 1)).sum().backward()
        optimizer.step()

        # Upload trained model
        print(f"{time_string()}: Uploading trained model to server.")
        if not openfed_api.upload():
            print("Uploading failed.")
            break
        print(f"{time_string()}: Uploaded!")

# >>> Finished
openfed_api.finish()

print(f"Finished.\nExit Client @{openfed_api.nick_name}.")