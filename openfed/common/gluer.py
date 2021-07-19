from typing import List

from openfed.utils import openfed_class_fmt, convert_to_list
from typing import Dict, Callable

class Gluer(object):
    def __init__(self):
        pass

    def __str__(self):
        return openfed_class_fmt.format(
            class_name='Gluer',
            description=''
        )


def glue(inst_a,
         inst_b,
         parall_func_list: List[str]           = None,
         parall_func_dict: Dict[str, Callable] = None):
    """Glue inst_a of TypeA with inst_b of TypeB, return a new inst_c of TypeC.
    Args:
        inst_a: Instance.
        inst_b: Instance.
        parall_func_dict: The parallel function dict. If value is None, use default one.
    ..note:: 
        If parall_func_dict is not None, we will rewrite the function in TypeA and TypeB. 
    In the new function, it will call TypeA and then TypeB. (make sure the function name
    is in both TypeA and TypeB). We will return A's output if it is not None, otherwise B's 
    output. If both A's output and B's output are dict, we will merge them.

    ..note::
        The variable in inst_a and inst_b will be merged obey the following rules:
        1. If variable in a and variable in b have the same name:
            use variable in a, if variable is not a dict.
            otherwise, a.update(b)
    """
    if parall_func_dict is None:
        parall_func_dict = {}

    parall_func_list = convert_to_list(parall_func_list)
    if parall_func_list is not None:
        parall_func_dict.update(
            {k: None for k in parall_func_list})  # type: ignore

    TypeA     = type(inst_a)
    TypeB     = type(inst_b)
    func_dict = dict()
    if parall_func_dict is not None:
        for func_name, func_impl in parall_func_dict.items():
            assert hasattr(TypeA, func_name), 'parall_func must in TypeA'
            assert hasattr(TypeB, func_name), 'parall_func must in TypeB'
            if func_impl is None:
                def parall_func(func_a, func_b):
                    def _parall_func(self, *args, **kwargs):
                        output_a = func_a(self, *args, **kwargs)
                        output_b = func_b(self, *args, **kwargs)
                        if output_a is None:
                            return output_b
                        if isinstance(output_a, dict) and isinstance(output_b, dict):
                            output_a.update(output_b)
                            return output_a
                        return output_a
                    return _parall_func
                func_dict[func_name] = parall_func(
                    getattr(TypeA, func_name), getattr(TypeB, func_name))
            else:
                func_dict[func_name] = func_impl

    name = f"Gluer_{inst_a.__class__.__name__}_{inst_b.__class__.__name__}"

    # Gluer make it enable to build the TypeC without any parameters.
    TypeC = type(name, (Gluer, TypeA, TypeB), func_dict)
    inst_c = TypeC()

    inst_c.__dict__.update(inst_a.__dict__)
    skip_keys = []
    for k, v in inst_b.__dict__.items():
        if k in inst_c.__dict__:
            if isinstance(v, dict) and isinstance(inst_c.__dict__[k], dict):
                inst_c.__dict__[k].update(v)
            else:
                skip_keys.append(k)
        else:
            inst_c.__dict__[k] = v
    if len(skip_keys):
        print(
            f"The following variables in {inst_b} have been written: {skip_keys}")
    return inst_c
