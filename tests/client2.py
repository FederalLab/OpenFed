import os
import sys
sys.path.insert(0, '/Users/densechen/code/OpenFed')
from openfed.core import Maintainer, World

world = World(role='openfed_follower')
maintainer = Maintainer(world=world, address_file='tests/client2.json')

print('Connected.')