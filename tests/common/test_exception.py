from openfed.common.exception import *


def test_exception():
    try:
        raise OpenFedException()
    except OpenFedException as e:
        print(e)

    try:
        raise AccessError()
    except AccessError as e:
        print(e)

    try:
        raise ConnectionNotBuild()
    except ConnectionNotBuild as e:
        print(e)
