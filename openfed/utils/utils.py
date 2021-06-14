from enum import Enum


def to_enum(value, enum_type: Enum):
    for enum in enum_type:
        if enum.value == value:
            return enum
    else:
        raise ValueError(f"{value} is not a valid enum {enum_type}")

def in_enum(value, enum_type: Enum):
    for enum in enum_type:
        if enum.value == value:
            return True
    else:
        return False
