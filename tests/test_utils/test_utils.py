from openfed.utils.utils import FMT, seed_everything, time_string


def test_time_string():
    time_string()


def test_seed_everything():
    seed_everything(0)


def test_openfed_title():
    print(FMT.openfed_title)
    print(
        FMT.openfed_class_fmt.format(
            class_name='test', description='test description'))
