import os
import sys
sys.path.insert(0, '/Users/densechen/code/OpenFed')
from openfed.core import FederatedGroup, World

world = World(role='openfed_leader', dal=True)
federated_group = FederatedGroup(world=world, address_file='tests/server.json')

while True:
    if len(federated_group.finished_queue) == 2:
        federated_group.manual_stop()
        print('Connected.')
        break