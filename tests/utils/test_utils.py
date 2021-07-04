from openfed.utils.utils import *


def test_utils():
    time_string()
    print(openfed_class_fmt.format(class_name=1, description=2))
    seed_everything(1)
    assert convert_to_list(None) == None
