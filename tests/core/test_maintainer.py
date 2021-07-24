from openfed.core import follower, leader, World, Maintainer
from openfed.common import load_address_from_file
import time
import os

def test_maintainer_leader():
    os.remove('/tmp/openfed.test.1')
    os.remove('/tmp/openfed.test.2')
    world = World(role=leader)
    maintainer = Maintainer(world=world, address_file='tests/core/leader_address.json')
    while len(maintainer.pending_queue):
        time.sleep(0.1)
    assert len(maintainer.finished_queue) == 3

def test_maintainer_follower_1():
    world = World(role=follower)
    address = load_address_from_file('tests/core/follower_address.json')[0]
    maintainer = Maintainer(world=world, address=address)
    while len(maintainer.pending_queue):
        time.sleep(0.1)
    assert len(maintainer.finished_queue) == 1

def test_maintainer_follower_2():
    world = World(role=follower)
    address = load_address_from_file('tests/core/follower_address.json')[1]
    maintainer = Maintainer(world=world, address=address)
    while len(maintainer.pending_queue):
        time.sleep(0.1)
    assert len(maintainer.finished_queue) == 1

def test_maintainer_follower_3():
    world = World(role=follower)
    address = load_address_from_file('tests/core/follower_address.json')[2]
    maintainer = Maintainer(world=world, address=address)
    while len(maintainer.pending_queue):
        time.sleep(0.1)
    assert len(maintainer.finished_queue) == 1