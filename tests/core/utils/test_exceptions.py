from openfed.core.utils.exceptions import *


def test_exception():
    try:
        raise ConnectTimeout()
    except ConnectTimeout as e:
        print(e)

    try:
        raise InvalidStoreReading()
    except InvalidStoreReading as e:
        print(e)

    try:
        raise InvalidStoreWriting()
    except InvalidStoreWriting as e:
        print(e)

    try:
        raise BuilddeliveryFailed()
    except BuilddeliveryFailed as e:
        print(e)

    try:
        raise DeviceOffline()
    except DeviceOffline as e:
        print(e)

    try:
        raise WrongState()
    except WrongState as e:
        print(e)
