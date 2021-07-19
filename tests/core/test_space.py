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
    assert not country.is_initialized()
    
    handler = country.init_process_group(
        backend='gloo',
        init_method='tcp://localhost:1996', 
        world_size=2,
        rank=0, 
        store=None,
        group_name='test_country',
    )

    while not handler():
        time.sleep(0.1)
    else:
        assert True

    assert country.is_initialized()

    assert country.get_rank() == 0
    assert country.get_world_size() == 2

    country.build_point2point_group(rank=0)

    country.destroy_process_group()
    assert not country.is_initialized()


def test_country_follower():
    world = World(role=follower)

    country = Country(world)
    assert not country.is_initialized()
    
    handler = country.init_process_group(
        backend='gloo',
        init_method='tcp://localhost:1996', 
        world_size=2,
        rank=1, 
        store=None,
        group_name='test_country',
    )

    while not handler():
        time.sleep(0.1)
    else:
        assert True

    assert country.is_initialized()

    assert country.get_rank() == 1
    assert country.get_world_size() == 2

    country.build_point2point_group(rank=0)

    country.destroy_process_group()
    assert not country.is_initialized()