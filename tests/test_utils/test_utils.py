from openfed.utils.utils import (openfed_class_fmt, openfed_title,
                                 seed_everything, time_string)


def test_time_string():
    time_string()


def test_seed_everything():
    seed_everything(0)


def test_openfed_title():
    print(openfed_title)
    print(
        openfed_class_fmt.format(
            class_name='test', description='test description'))
