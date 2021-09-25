# @Author            : FederalLab
# @Date              : 2021-09-25 16:57:05
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:57:05
# Copyright (c) FederalLab. All rights reserved.
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
