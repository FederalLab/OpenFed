import torch
import torch.nn as nn
import torch.optim as optim
import random
import openfed
from openfed.optim.elastic_aux import ElasticAux
import time
import openfed.aggregate as aggregate


parser = openfed.parser
args = parser.parse_args()
if args.rank == 0:
    print("Waiting for client to connect...")
    openfed_api = openfed.Backend(address=openfed.Address(args=args))
else:
    print(f"Connect to server @ rank{args.rank}")
    openfed_api = openfed.Frontend(address=openfed.Address(args=args))

openfed_api.openfed_digest()

# Build a Model
net = nn.Linear(1, 1)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# for server
aggregator = aggregate.ElasticAggregator(
    net.parameters(), other_keys="momentum_buffer")

# for client
elastic_aux = ElasticAux(net.parameters())

# tell model to the backend
openfed_api.set_state_dict(net.state_dict(keep_vars=True))

# tell optimizer and aggregator to backend
if args.rank == 0:
    openfed_api.set_optimizer(optimizer)
    openfed_api.set_aggregator(aggregator)

    openfed_api.run()
    openfed_api.finish()
    print("Finished. Exit Server.")
    exit(0)

for i in range(1, 55):
    print(f"Train @{i}")
    # Download a new model
    if not openfed_api.download():
        break

    time.sleep(0.1)

    # Traning
    net(torch.randn(128, 1, 1)).sum().backward()
    elastic_aux.step()
    optimizer.step()

    # Submitted
    openfed_api.pack_state(optimizer, keys=['momentum_buffer'])
    openfed_api.pack_state(elastic_aux)

    openfed_api.set_task_info({"train_instances": random.randint(1, 200)})

    if not openfed_api.upload():
        break

openfed_api.finish()
print(f"Finished. Exit Client @{args.rank}.")
