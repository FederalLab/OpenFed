import random

# >>> Import OpenFed
import openfed
import torch
import torch.nn as nn
import torch.optim as optim
from openfed.api import StepAt
from openfed.container import AverageAgg
from openfed.utils import time_string


def test_follower():
    # >>> set log level
    openfed.logger.log_level(level="DEBUG")

    # >>> Get default arguments from OpenFed
    args = openfed.parser.parse_args()
    args.rank = 1

    # >>> Specify an API for building federated learning
    openfed_api = openfed.API(frontend=args.rank > 0)

    # >>> Register more step functions.
    # You can register a step function to openfed_api like following:
    # stop_at_version = openfed.StopAtVersion(max_version=10)
    # openfed_api.register_step(stop_at_version)
    # Or use the with context to add a sequence of step function to openfed_api automatically.
    with StepAt(openfed_api):
        openfed.api.AggregateCount(
            count=args.world_size-1, checkpoint="/tmp/openfed-model")
        openfed.api.StopAtVersion(max_version=3)
        openfed.api.AfterDownload()
        openfed.api.BeforeUpload()

    # >>> Connect to Address.
    openfed_api.build_connection(address=openfed.Address(args=args))

    # Build Network
    net = nn.Linear(1, 1)

    # Define optimizer (use the same optimizer in both server and client)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    # Define agg (actually, this is only used for server end): FedAvg, ElasticAgg
    agg = AverageAgg(net.parameters())

    # >>> Set optimizer and agg for federated learning.
    openfed_api.set_aggregator_and_optimizer(agg, optimizer)

    # >>> Tell OpenFed API which data should be transferred.
    openfed_api.set_state_dict(net.state_dict(keep_vars=True))

    # Context `with openfed_api` will go into the specified settings about openfed_api.
    # Otherwise, will use the default one which shared by global OpenFed world.
    with openfed_api:

        # >>> If openfed_api is a backend, call `run()` will go into the loop ring.
        # >>> Call `start()` will run it as a thread.
        # >>> If openfed_api is a frontend, call `run()` will directly skip this function automatically.
        openfed_api.backend_loop()

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

            # >>> Update inner model version
            openfed_api.update_version()

            # Upload trained model
            print(f"{time_string()}: Uploading trained model to server.")
            if not openfed_api.upload():
                print("Uploading failed.")
                break
            print(f"{time_string()}: Uploaded!")

    # >>> Finished
    openfed_api.finish()

    print(f"Finished.\nExit Client @{openfed_api.nick_name}.")


if __name__ == "__main__":
    test_follower()