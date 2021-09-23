# Copyright (c) FederalLab. All rights reserved.
from typing import Any, List, Union

from prettytable import PrettyTable


def string_trim(string: Union[str, Any], length: int = 15):
    r"""Converts a string or others to a trimmed string, whose length is not longer
    than ``length``.

    Args:
        string: A string to trim.
        length: The length of the string.

    Returns:
        The trimmed string.

    Example::

        >>> from openfed.utils import string_trim
        >>> string_trim(15, 4)
        '15'
        >>> string_trim('this is a message from openfed', 15)
        'this is...openfed'
    """
    # make sure string is string
    if isinstance(string, float):
        string = f'{string:.2f}'
    else:
        string = str(string)
    if len(string) > length + 3:
        half = length // 2
        return string[:half] + '...' + string[len(string) - half:]
    else:
        return string


def _tablist(head: List[Any],
             data: List[Any],
             multi_rows: bool = False) -> str:
    columns = 80
    length = (columns - len(head) * 3 - 1) // len(head)
    head = [string_trim(h, length) for h in head]
    # String operation may string the head with the same result.
    # In this case, you should split then into more tables,
    # or just use a different head.
    # By the way, the head should with different starts, which
    # can largely avoid this error.
    assert len(head) == len(
        set(head)), 'String trim operation make some head with the some name.'
    table = PrettyTable(head)
    if multi_rows:
        for d in data:
            table.add_row([string_trim(_d, length) for _d in d])
    else:
        table.add_row([string_trim(d, length) for d in data])

    return str(table)


def tablist(head: List[Any],
            data: List[Any],
            items_per_row: int = 8,
            force_in_one_row: bool = False) -> str:
    """If len(head) > items_per_row, we will split into multi-tables.

    .. note::
        If `force_in_one_row` is ``True``, `items_per_row` will be ignored.

    Args:
        head: Head of data.
        data: List of data, must be the same length as the head.
        items_per_row: Number of items per row.
        force_in_one_row: Show all data in one row.

    Examples::

        >>> from openfed.utils import tablist
        >>> head = ['a', 'b', 'c', 'd', 'e', 'f']
        >>> data = [1, 2, 3, 4, 5, 6]
        >>> print(tablist(head, data, 3))
        +---+---+---+
        | a | b | c |
        +---+---+---+
        | 1 | 2 | 3 |
        +---+---+---+
        +---+---+---+
        | d | e | f |
        +---+---+---+
        | 4 | 5 | 6 |
        +---+---+---+
        >>> print(tablist(head, data, force_in_one_row=True))
        +---+---+---+---+---+---+
        | a | b | c | d | e | f |
        +---+---+---+---+---+---+
        | 1 | 2 | 3 | 4 | 5 | 6 |
        +---+---+---+---+---+---+
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        >>> print(tablist(head, data, force_in_one_row=True))
        +---+---+---+----+----+----+
        | a | b | c | d  | e  | f  |
        +---+---+---+----+----+----+
        | 1 | 2 | 3 | 4  | 5  | 6  |
        | 7 | 8 | 9 | 10 | 11 | 12 |
        +---+---+---+----+----+----+
    """
    table_list = []
    if force_in_one_row:
        i = 0
        rows = []
        while i < len(data):
            if i + len(head) <= len(data):
                rows.append(data[i:i + len(head)])
            else:
                rows.append(data[i:] + [' '] * len(data) - i)
            i += len(head)
        table_list.append(_tablist(head, rows, multi_rows=True))
    else:
        i = 0
        while i < len(head):
            if i + items_per_row < len(head):
                table_list.append(
                    _tablist(head[i:i + items_per_row],
                             data[i:i + items_per_row]))
            else:
                table_list.append(_tablist(head[i:], data[i:]))
            i += items_per_row
    return '\n'.join(table_list)
