from numbers import Number
from typing import Any, Dict, List, Union

import numpy as np


class Initializer:
    @property
    def config(self) -> Dict[str, Any]:
        return {}


class ConstantInitializer(Initializer):
    def __init__(self, value: Number = 0.0) -> None:
        self.value = value

    @property
    def config(self) -> Dict[str, Any]:
        return dict(value=self.value)


class UniformInitializer(Initializer):
    def __init__(self, a: float = 0.0, b: float = 1.0) -> None:
        if a >= b:
            raise ValueError("The minimum should be strictly less than the maximum")
        self.a = a
        self.b = b

    @property
    def config(self) -> Dict[str, Any]:
        return dict(a=self.a, b=self.b)


class NormalInitializer(Initializer):
    def __init__(self, mean: float = 0.0, stddev: float = 1.0) -> None:
        if stddev <= 0.0:
            raise ValueError("The standard deviation should be a positive number")
        self.mean = mean
        self.stddev = stddev

    @property
    def config(self) -> Dict[str, Any]:
        return dict(mean=self.mean, stddev=self.stddev)


class DirichletInitializer(Initializer):
    def __init__(self, alpha: Union[float, List[float]] = 1.0, *, axis: int = -1) -> None:
        if isinstance(alpha, list):
            if any(a < 0.0 for a in alpha) or not np.isclose(sum(alpha), 1.0):
                raise ValueError(
                    "The concentration parameters should be non-negative and sum to one"
                )
        self.alpha = alpha
        self.axis = axis

    @property
    def config(self) -> Dict[str, Any]:
        return dict(alpha=self.alpha, axis=self.axis)
