import torch
import openfed.federated as fed
import os

fed_world = fed.FederatedWorld()
fed_world.init_process_group('gloo', init_method='tcp://127.0.0.1:23456', rank=rank, world_size=3)
fed.register.register_federated_world(fed_world)

linear = torch.nn.Linear(1, 1)
state_dict = linear.state_dict()

fed_world.build_point2point_group(rank=0)