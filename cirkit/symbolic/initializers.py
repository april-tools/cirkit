from abc import ABC
from numbers import Number
from typing import Any, Dict, List, Union

import numpy as np


class Initializer(ABC):
    """The abstract symbolic initializer class. Symbolic initializers are usually assigned to
    [TensorParameter][cirkit.symbolic.parameters.TensorParameter] upon their instantiation."""

    @property
    def config(self) -> Dict[str, Any]:
        """Retrieves the hyperparameters of the initializer.

        Returns:
            A dictionary mapping hyperparameter names to their values.
        """
        return {}


class ConstantInitializer(Initializer):
    """A symbolic constant initializer, which initializes all the entries of a tensor with the
    same scalar value."""

    def __init__(self, value: Number = 0.0) -> None:
        """Initializes a constant initializer.

        Args:
            value: The value used for initialization, it can be either an integer, a real number
                or a complex number.

        Raises:
            ValueError: If the initiaization value is not a number.
        """
        if not isinstance(value, Number):
            raise ValueError("The value must be a number")
        self.value = value

    @property
    def config(self) -> Dict[str, Any]:
        return {"value": self.value}


class ConstantTensorInitializer(Initializer):
    """A symbolic constant tensor initializer, which initializes a tensor using a given
    Numpy array of the same shape."""

    def __init__(self, value: np.ndarray) -> None:
        """Initializes a constant tensor initializer.

        Args:
            value: The value used for initialization, which must be a Numpy array.

        Raises:
            ValueError: If the initiaization value is not a Numpy array.
        """
        if not isinstance(value, np.ndarray):
            raise ValueError("The value must be a Numpy array")
        self.value = value

    @property
    def config(self) -> Dict[str, Any]:
        return {"value": self.value}


class UniformInitializer(Initializer):
    """A symbolic uniform initializer, which initializes all the entries of a tensor
    by sampling independently from a univariate uniform distribution."""

    def __init__(self, a: float = 0.0, b: float = 1.0) -> None:
        """Initializes a uniform initializer, given minimum and maximum.

        Args:
            a: The minimum.
            b: The maximum.

        Raises:
            ValueError: If the minimum is not strictly less than the maximum.
        """
        if a >= b:
            raise ValueError("The minimum should be strictly less than the maximum")
        self.a = a
        self.b = b

    @property
    def config(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b}


class NormalInitializer(Initializer):
    """A symbolic normal initializer, which initializes all the entries of a tensor
    by sampling independently from a univariate normal distribution."""

    def __init__(self, mean: float = 0.0, stddev: float = 1.0) -> None:
        """Initializes a normal initializer, given mean and standard deviation.

        Args:
            mean: The mean.
            stddev: The standard deviation.

        Raises:
            ValueError: If the standard deviation is not a positive number.
        """
        if stddev <= 0.0:
            raise ValueError("The standard deviation should be a positive number")
        self.mean = mean
        self.stddev = stddev

    @property
    def config(self) -> Dict[str, Any]:
        return {"mean": self.mean, "stddev": self.stddev}


class DirichletInitializer(Initializer):
    """A symbolic Dirichlet initializer, which initializes all the entries of a tensor
    along one axis by sampling independently from a Dirichlet distribution."""

    def __init__(self, alpha: Union[float, List[float]] = 1.0, *, axis: int = -1) -> None:
        """Initializes a Dirichlet initializer, given the concentration parameters
        and the axis along which the sampled values will sum to one.

        Args:
            alpha: The concentration parameter. If it is a list, then different concentrations
                will be used for each random variable being sampled.
            axis: The axis along which the sampled values will sum to one.

        Raises:
            ValueError: If the concentration parameter is not positive or if it contains
                non-positve values.
        """
        if not isinstance(alpha, (float, list)):
            raise ValueError("The concentration parameters should be either a scalar or a list")
        if (isinstance(alpha, float) and alpha <= 0.0) or (
            isinstance(alpha, list) and any(a <= 0.0 for a in alpha)
        ):
            raise ValueError("The concentration parameters should be positive")
        self.alpha = alpha
        self.axis = axis

    @property
    def config(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "axis": self.axis}
