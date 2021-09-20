# Copyright (c) FederalLab. All rights reserved.
from typing import Any, List

from prettytable import PrettyTable


def _string_trim(string: str, length: int = 15):
    # make sure string is string
    if isinstance(string, int):
        string = f"{string}"
    elif isinstance(string, float):
        string = f"{string:.2f}"
    else:
        string = str(string)
    if len(string) > length + 3:
        return string[:length] + "..."
    else:
        return string


def _tablist(head: List[Any],
             data: List[Any],
             multi_rows: bool = False) -> str:
    columns = 80
    length = (columns - len(head) * 3 - 1) // len(head)
    head = [_string_trim(h, length) for h in head]
    # String operation may string the head with the same result.
    # In this case, you should split then into more tables,
    # or just use a different head.
    # By the way, the head should with different starts, which
    # can largely avoid this error.
    assert len(head) == len(
        set(head)), "String trim operation make some head with the some name."
    table = PrettyTable(head)
    if multi_rows:
        for d in data:
            table.add_row([_string_trim(_d, length) for _d in d])
    else:
        table.add_row([_string_trim(d, length) for d in data])

    return str(table)


def tablist(head: List[Any],
            data: List[Any],
            items_per_row: int = 8,
            force_in_one_row: bool = False) -> str:
    """
        If len(head) > items_per_row, we will split into multi-tables.
        If force_in_one_row is True, items_per_row will be ignored.
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
    return "\n".join(table_list)
