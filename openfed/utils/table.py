from typing import Any, List

from prettytable import PrettyTable
import os


def _string_trim(string: str, length: int = 15):
    # make sure string is string
    if isinstance(string, int):
        string = f"{string}"
    elif isinstance(string, float):
        string = f"{string:.2f}"
    else:
        string = str(string)
    if len(string) > length + 3:
        return string[:length]+"..."
    else:
        return string


def _tablist(head: List[Any], data: List[Any]) -> str:
    rows, columns = os.popen('stty size', 'r').read().split()
    length = (int(columns) - len(head) * 3 - 1) // len(head)
    table = PrettyTable([_string_trim(h, length) for h in head])
    table.add_row([_string_trim(d, length) for d in data])

    return str(table)


def tablist(head: List[Any], data: List[Any], items_per_row: int = 4, force_in_one_row: bool = False) -> str:
    """
        If len(head) > items_per_row, we will split into multi-tables.
        If force_in_one_row is True, items_per_row will be ignored.
    """
    table_list = []
    if force_in_one_row:
        table_list.append(_tablist(head, data))
    else:
        i = 0
        while i < len(head):
            if i + items_per_row < len(head):
                table_list.append(
                    _tablist(head[i:i+items_per_row], data[i:i+items_per_row]))
            else:
                table_list.append(_tablist(head[i:], data[i:]))
            i += items_per_row
    return "\n".join(table_list)
