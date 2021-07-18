from openfed.core.space import *

def test_world():
    world_leader = World(role=leader)

    world_follower = World(role=follower)

    assert world_leader.leader
    assert not world_leader.follower

    assert world_follower.follower
    assert not world_follower.leader

    assert world_leader.default_delivery == None
    assert world_leader.default_pg == None

    print(world_leader)
    print(world_follower)

    # try to kill world
    world_leader.kill()
    world_follower.kill()


def test_country_leader():
    world = World(role=leader)

    country = Country(world)
    
    country.init_process_group(
        backend='gloo',
        init_method='tcp://', 
        world_size=1,
        rank=1, 
        store=None,
        group_name='',
    )

def test_country_follower():
    pass
