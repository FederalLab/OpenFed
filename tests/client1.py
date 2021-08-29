import os
import sys
sys.path.insert(0, '/Users/densechen/code/OpenFed')
from openfed.core import FederatedGroup, World

world = World(role='openfed_follower')
federated_group = FederatedGroup(world=world, address_file='tests/client1.json')

print('Connected.')