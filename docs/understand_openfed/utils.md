# Utils

This component provides some useful functions, such as `seed_everything`, `time_string`.

## Format output as table

Sometimes, we can receive a better visualization via show some data in a table.

You can do this with :func:`tablist`:

```shell
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
```
