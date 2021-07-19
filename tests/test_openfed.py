# import os
# import sys
# sys.path.insert(0, "/Users/densechen/code/OpenFed")

# import random

# # >>> Import OpenFed
# import openfed
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from openfed.container import AverageAgg
# from openfed.utils import time_string


# def main(args):
#     # >>> set log level
#     openfed.logger.log_level(level="DEBUG")

#     # Build Network
#     net = nn.Linear(1, 1)
#     # Move to GPU
#     net.to(args.device)

#     # Define optimizer (use the same optimizer in both server and client)
#     optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

#     # Define agg (actually, this is only used for server end): FedAvg, ElasticAgg
#     aggregator = AverageAgg(net.parameters())

#     # >>> Specify an API for building federated learning
#     openfed_api = openfed.API(
#         args.fed_rank > 0, 
#         net.state_dict(keep_vars=True), 
#         optimizer, 
#         aggregator)
#     openfed_api.max_try_times = args.max_try_times

#     # >>> Connect to Address.
#     openfed_api.build_connection(address=openfed.Address(args=args))

#     # >>> Register more step functions.
#     # You can register a step function to openfed_api like following:
#     # stop_at_version = openfed.StopAtVersion(max_version=10)
#     # openfed_api.register_step(stop_at_version)
#     # Or use the with context to add a sequence of step function to openfed_api automatically.
#     with openfed_api:
#         openfed.api.Aggregate(
#             count=args.fed_world_size-1, checkpoint="/tmp/openfed-model")
#         openfed.api.Terminate(max_version=3)
#         openfed.api.Download()
#         openfed.api.Upload()

#         # >>> If openfed_api is a backend, call `run()` will go into the loop ring.
#         # >>> Call `start()` will run it as a thread.
#         # >>> If openfed_api is a role, call `run()` will directly skip this function automatically.
#         if not openfed_api.backend_loop():
#             # Do simulation random times at [10, 70].
#             for i in range(1, random.randint(10, 70)):
#                 print(f"{time_string()}: Simulation @{i}")

#                 # Download latest model.
#                 print(f"{time_string()}: Downloading latest model from server.")
#                 if not openfed_api.transfer(to=False):
#                     print(f"Downloading failed.")
#                     break

#                 # Downloaded
#                 print(f"{time_string()}: Downloaded!")

#                 # Start a standard forward/backward pass.
#                 optimizer.zero_grad()
#                 net(torch.randn(128, 1, 1).to(args.device)).sum().backward()
#                 optimizer.step()

#                 # >>> Update inner model version
#                 openfed_api.update_version()

#                 # Upload trained model
#                 print(f"{time_string()}: Uploading trained model to server.")
#                 if not openfed_api.transfer(to=True):
#                     print("Uploading failed.")
#                     break
#                 print(f"{time_string()}: Uploaded!")

#     print(f"Finished.\nExit Client @{openfed_api.nick_name}.")
#     # >>> Finished
#     openfed_api.finish(auto_exit=True)


# if __name__ == "__main__":
#     # >>> Get default arguments from OpenFed
#     openfed.parser.add_argument("--max_try_times", default=15, type=int)
#     openfed.parser.add_argument("--gpu", default=False, action='store_true')
#     args = openfed.parser.parse_args()
#     if torch.cuda.is_available() and args.gpu:
#         args.gpu = args.fed_rank % torch.cuda.device_count()
#         args.device = torch.device(args.gpu)
#         torch.set_device(args.gpu)
#     else:
#         args.device = torch.device("cpu")
#     print(f"Use {args.device}.")
#     main(args)

def test_leader():
    print("Leader")

def test_follower():
    print("Follower")