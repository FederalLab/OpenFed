from openfed.common.array import Array
from threading import Lock


def test_array():
    # Create an array.
    default_mapping = dict()
    array = Array(default_mapping, Lock())

    assert len(array) == 0

    # add a item to mapping.
    default_mapping["test_array"] = 'OK'

    assert len(array) == 1

    assert array.default_key == 'test_array'
    assert array.default_value == 'OK'

    for k, v in array:
        print(k, v)

    key, value = array[0]

    assert key == 'test_array'
    assert value == 'OK'
