# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import warnings
from typing import Any, Callable, Dict
from openfed.utils import openfed_class_fmt


class Gluer(object):
    """An empty class.
    """

    def __init__(self):
        pass

    def __str__(self):
        return openfed_class_fmt.format(
            class_name  = 'Gluer',
            description = self.__class__.__name__,
        )


def glue(inst_a: Any,
         inst_b: Any,
         extra_func: Dict[str, Callable] = None):
    """Glue ``inst_a`` of ``TypeA`` with ``inst_b`` of ``TypeB``, 
    return a new ``inst_c`` of ``TypeC``. At default, ``inst_c`` will keep variables and 
    functions from  ``inst_a`` if they appeared both in ``inst_a`` and ``inst_b``.

    Args:
        inst_a: Instance of TypeA.
        inst_b: Instance of TypeB.
        extra_func: The new function registered to TypeC.

    .. note::
        1. Dictionary variables will be re-assigned via `inst_c.v = inst_a.v.update(inst_b.v)`.
        2. If extra_func's value is not provided (`None`), a new function will created via
            `new_f = lambda: inst_a.func() or inst_b.func()`.
        3. The variable starts with `_` or `__` will be skipped automatically in `inst_b`.

    .. Example::
        >>> class TypeA(object):
        ...     def print_A(self):
        ...             print('print_A')
        ...     def print(self):
        ...             print('print_A')
        ...     def pprint(self):
        ...             print('print_A')
        ...             return {'pprint_A': 'pprint_A'}
        ...     def __init__(self):
        ...             self.name = 'TypeA'
        ...             self.name_dict = {'TypeA_name': 'TypeA'}
        ... 
        >>> class TypeB(object):
        ...     def print_B(self):
        ...             print('print_B')
        ...     def print(self):
        ...             print('print_B')
        ...     def pprint(self):
        ...             print('print_B')
        ...             return {'pprint_B': 'pprint_B'}
        ...     def __init__(self):
        ...             self.name = 'TypeB' 
        ...             self.name_dict = {'TypeB_name': 'TypeB'}
        ... 
        >>> extra_func = dict(
        ...     pprint=None,
        ...     print_C=lambda self:print("Type_C")
        ... )
        >>> inst_a = TypeA()
        >>> inst_b = TypeB()
        >>> inst_c = glue(inst_a, inst_b, extra_func)
        UserWarning: 1 variables of <__main__.TypeB object at 0x7f7ee00a5910> are discarded, 1 variables of <__main__.TypeB object at 0x7f7ee00a5910> are merged.
        UserWarning: Discarded keys: ['name']
        UserWarning: Merged keys: ['name_dict']
        >>> inst_c.print()
        print_A
        >>> inst_c.pprint()
        print_A
        print_B
        {'pprint_A': 'pprint_A', 'pprint_B': 'pprint_B'}
        >>> inst_c.name
        'TypeA'
        >>> inst_c.name_dict
        {'TypeA_name': 'TypeA', 'TypeB_name': 'TypeB'}
        >>> inst_c.print_C()
        Type_C
    """
    extra_func = extra_func or {}
    TypeA = type(inst_a)
    TypeB = type(inst_b)
    func_dict = dict()

    for func_name, func_impl in extra_func.items():
        if func_impl is not None:
            func_dict[func_name] = func_impl
        else:
            if not(hasattr(TypeA, func_name) and hasattr(TypeB, func_name)):
                raise RuntimeError(
                    f"{TypeA} and {TypeB} must provide the implementation of {func_name}.")

            def glue_func(func_a, func_b):
                # Create a decorator that glue func_a and func_b.
                def _glue_func(*args, **kwargs):
                    # If the output is dictionary, we will return output_a.update(output_b)
                    # Otherwise, we only return `output_a or output_b`
                    output_a = func_a(*args, **kwargs)
                    output_b = func_b(*args, **kwargs)
                    if isinstance(output_a, dict) and isinstance(output_b, dict):
                        output_a.update(output_b)
                        return output_a
                    return output_a or output_b
                return _glue_func
            func_dict[func_name] = glue_func(
                func_a = getattr(TypeA, func_name),
                func_b = getattr(TypeB, func_name))

    name = f"Gluer_{inst_a.__class__.__name__}_{inst_b.__class__.__name__}"

    # Gluer make it enable to build the TypeC without any parameters.
    TypeC = type(name, (Gluer, TypeA, TypeB), func_dict)
    inst_c = TypeC()

    inst_c.__dict__.update(inst_a.__dict__)
    discard_keys  = []
    merge_keys = []
    for k, v in inst_b.__dict__.items():
        # The key starts with `_` or `__` will be skipped automatically.
        if not k.startswith("_"):
            if k in inst_c.__dict__:
                if isinstance(v, dict) and isinstance(inst_c.__dict__[k], dict):
                    inst_c.__dict__[k].update(v)
                    merge_keys.append(k)
                else:
                    discard_keys.append(k)
            else:
                inst_c.__dict__[k] = v

    warnings.warn(
        f"{len(discard_keys)} variables of {inst_b} are discarded,"
        f"{len(merge_keys)} variables of {inst_b} are merged.")

    if discard_keys:
        warnings.warn(f"Discarded keys: {discard_keys}")
    if merge_keys:
        warnings.warn(f"Merged keys: {merge_keys}")

    return inst_c
