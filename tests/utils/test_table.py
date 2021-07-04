from openfed.utils.table import tablist


def test_tablist():
    print(tablist(head=['a', 'b', 'c'], data=[1, 2, 3], force_in_one_row=True))
