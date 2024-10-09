from enum import IntEnum, auto
from numbers import Number

import numpy as np


class DataType(IntEnum):
    """The available symbolic data types. Note that these data types are precision-agnostic."""

    INTEGER = auto()
    """The integer numbers data type."""
    REAL = auto()
    """The real numbers data type."""
    COMPLEX = auto()
    """The complex numbers data type."""


def dtype_value(x: Number | np.ndarray) -> DataType:
    """Given a number or Numpy array, return its symbolic data type.

    Args:
        x: A number, which can be a Python integer, float or complex number.
            Alternatively, it can be a Numpy array.

    Returns:
        The symbolic data type associated to the given number or Numpy array.

    Raises:
        ValueError: If the given number is neither an integer, nor a float, nor a complex number,
            nor a Numpy array.
    """
    if isinstance(x, int):
        return DataType.INTEGER
    if isinstance(x, float):
        return DataType.REAL
    if isinstance(x, complex):
        return DataType.COMPLEX
    if isinstance(x, np.ndarray):
        if issubclass(x.dtype.type, np.integer):
            return DataType.INTEGER
        if issubclass(x.dtype.type, np.floating):
            return DataType.REAL
        if issubclass(x.dtype.type, np.complexfloating):
            return DataType.COMPLEX
    raise ValueError(f"Cannot retrieve data type of number of type {type(x)}")
