from openfed.common.base import peeper

def test_peeper():
    peeper.add_to_peeper('test', list())
    peeper.get_from_peeper('test')
    peeper.remove_from_peeper('test')

    assert 'test' not in peeper.obj_item_mapping


    print(peeper)