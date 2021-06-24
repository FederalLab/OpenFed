# OpenFed

## Features

1. Async/Sync Transfer Support
2. Dynamic Address Management
3. Unified Frontend and Backend API
4. Arbitrary Federated Topology Connection
5. Transfer Data without Hesitation
6. PyTorch Coding Style

## Install

Python>=3.7, PyTorch=1.9.0 are required.

```bash
conda create -n openfed python=3.7 -y
conda activate openfed
pip3 install -r requirements.txt

# test
python3 -m openfed.launch --nproc_per_node 3 --logdir /tmp --server_output demo.py

# make sure /tmp/openfed.sharefile does not exist before run this script.
python3 -m openfed.launch --nproc_per_node 11 --logdir /tmp --server_output demo.py --init_method file:///tmp/openfed.sharefile
```

## Build Federated Learning in an Unprecedented Simple Way

```python
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
```

**Run it as one server with two clients**:

```bash
# Start server on terminal 1
python demo.py --rank 0 --world_size 3
# Start client 1 on terminal 2
python demo.py --rank 1 --world_size 3
# Start client 2 on terminal 3
python demo.py --rank 2 --world_size 3
```


## Project Structure

```bash
openfed
├── __init__.py
├── aggregate
│   ├── __init__.py
│   ├── aggregator.py
│   ├── average.py
│   ├── elastic.py
│   └── naive.py
├── api.py
├── common
│   ├── __init__.py
│   ├── address.py
│   ├── array.py
│   ├── constants.py
│   ├── exception.py
│   ├── hook.py
│   ├── logging.py
│   ├── package.py
│   ├── parser.py
│   ├── peeper.py
│   ├── thread.py
│   ├── vars.py
│   └── wrapper.py
├── data
│   ├── __init__.py
│   ├── dataset.py
│   ├── nlp
│   │   ├── __init__.py
│   │   ├── shakespeare.py
│   │   └── stackoverflow.py
│   ├── partitioner.py
│   └── vision
│       ├── __init__.py
│       └── emnist.py
├── federated
│   ├── __init__.py
│   ├── country.py
│   ├── deliver
│   │   ├── __init__.py
│   │   ├── delivery.py
│   │   └── functional.py
│   ├── destroy.py
│   ├── functional.py
│   ├── inform
│   │   ├── __init__.py
│   │   ├── functional.py
│   │   └── informer.py
│   ├── joint.py
│   ├── lock.py
│   ├── maintainer.py
│   ├── register.py
│   ├── reign.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── exception.py
│   │   └── utils.py
│   └── world.py
├── helper.py
├── launch.py
├── optim
│   ├── __init__.py
│   └── elastic_aux.py
├── unified
│   ├── __init__.py
│   ├── backend.py
│   ├── frontend.py
│   ├── step_hooks
│   │   ├── __init__.py
│   │   ├── step_after_destroy.py
│   │   ├── step_after_download.py
│   │   ├── step_after_upload.py
│   │   ├── step_at_failed.py
│   │   ├── step_at_invalid_state.py
│   │   ├── step_at_last.py
│   │   ├── step_at_new_episode.py
│   │   ├── step_at_zombie.py
│   │   ├── step_before_destroy.py
│   │   ├── step_before_download.py
│   │   └── step_before_upload.py
│   └── unify.py
└── utils
    ├── __init__.py
    ├── keyboard.py
    ├── table.py
    └── utils.py

13 directories, 71 files
```
