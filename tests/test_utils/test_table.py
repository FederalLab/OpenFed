# @Author            : FederalLab
# @Date              : 2021-09-25 16:57:01
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-25 16:57:01
# Copyright (c) FederalLab. All rights reserved.
from openfed.utils.table import string_trim, tablist


def test_string_trim():
    string_trim('hello word', 8)
    string_trim(1.22222222, 4)


def test_tablist():

    head = ['a', 'b', 'c', 'd', 'e', 'f']
    data = [1, 2, 3, 4, 5, 6]

    tablist(head, data, items_per_row=3)
    tablist(head, data, items_per_row=10)
    tablist(head, data, force_in_one_row=True)
