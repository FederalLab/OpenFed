from openfed.common.clone import Clone


class MyClass(Clone):
    def __init__(self):
        self.list = []
        self.number = 0


def test_clone():
    # Create a class
    my_class = MyClass()
    clone_of_my_class = my_class.clone()

    assert id(my_class.list) == id(clone_of_my_class.list)

    my_class.number = 1

    assert my_class.number == 1
    assert clone_of_my_class.number == 0
