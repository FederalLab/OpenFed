import os
import sys
sys.path.insert(0, '/Users/densechen/code/OpenFed')
from openfed.core import Maintainer, World

world = World(role='openfed_leader', async_op='false', dal=True)
maintainer = Maintainer(world=world, address_file='tests/server.json')

while True:
    if len(maintainer.finished_queue) == 2:
        maintainer.manual_stop()
        print('Connected.')
        break