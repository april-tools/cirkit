from enum import IntEnum, auto
from numbers import Number


class DataType(IntEnum):
    INTEGER = auto()
    REAL = auto()
    COMPLEX = auto()


def dtype_value(x: Number) -> DataType:
    if isinstance(x, int):
        return DataType.INTEGER
    if isinstance(x, float):
        return DataType.REAL
    if isinstance(x, complex):
        return DataType.COMPLEX
    raise ValueError(f"Cannot retrieve data type of number of type {type(x)}")
