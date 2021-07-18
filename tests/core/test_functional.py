from glob import glob
from openfed.core.functional import *
from openfed.core.space import *

def test_functional_leader():
    world = World(role=leader)

    country = Country(world)
    assert not country.is_initialized()
    
    handler = country.init_process_group(
        backend='gloo',
        init_method='file:///tmp/openfed.sharefile.test_functional', 
        world_size=2,
        rank=0, 
        store=None,
        group_name='test_country',
    )

    while not handler():
        time.sleep(0.1)
    else:
        assert True

    tensor = torch.zeros(1)
    handler = isend(
        tensor,
        dst=1,
        tag=0,
        country=country,
        global_rank=True,
    )
    handler.wait()

    tensor = torch.zeros(0)
    recv(tensor, src=1, country=country, global_rank=True)
    assert tensor.item() == 1

def test_functional_follower():
    world = World(role=follower)

    country = Country(world)
    assert not country.is_initialized()
    
    handler = country.init_process_group(
        backend='gloo',
        init_method='file:///tmp/openfed.sharefile.test_functional', 
        world_size=2,
        rank=1, 
        store=None,
        group_name='test_country',
    )

    while not handler():
        time.sleep(0.1)
    else:
        assert True

    tensor = torch.ones(1)
    handler = irecv(
        tensor,
        src=0,
        tag=0,
        country=country,
        global_rank=True,
    )
    handler.wait()
    assert tensor.item() == 0

    tensor = torch.ones(1)
    send(tensor, dst=0, country=country, global_rank=True)
