from abc import ABC, abstractmethod
from numbers import Number
from typing import Any

import numpy as np


class Initializer(ABC):
    """The abstract symbolic initializer class. Symbolic initializers are usually assigned to
    [TensorParameter][cirkit.symbolic.parameters.TensorParameter] upon their instantiation."""

    @property
    def config(self) -> dict[str, Any]:
        """Retrieves the hyperparameters of the initializer.

        Returns:
            A dictionary mapping hyperparameter names to their values.
        """
        return {}

    @abstractmethod
    def allows_shape(self, shape: tuple[int, ...]) -> bool:
        """Checks whether the given parameter shape is supported by the initializer.

        Args:
            shape: The parameter shape.

        Returns:
            True if the shape is supported, False otherwise.
        """


class ElementwiseInitializer(Initializer, ABC):
    """An elementwise initializer initializes a parameter tensor by setting each entry
    using the same function, such as by sampling independently from a univariate distribution.
    Therefore, an elementwise initializer allows any parameter shape by default.
    """

    def allows_shape(self, shape: tuple[int, ...]) -> bool:
        return True


class ConstantTensorInitializer(Initializer):
    """A symbolic constant initializer, which initializes all the entries of a tensor with the
    same value, which can be either a scalar or a Numpy array of the same shape."""

    def __init__(self, value: Number | np.ndarray) -> None:
        """Initializes a constant tensor initializer.

        Args:
            value: The value used for initialization, which must be either an integer,
                a real number, a complex number or a Numpy array.

        Raises:
            ValueError: If the initiaization value is not of the allowed types.
        """
        if not isinstance(value, (Number, np.ndarray)):
            raise ValueError("The value must be either a number or a Numpy array")
        self.value = value

    @property
    def config(self) -> dict[str, Any]:
        return {"value": self.value}

    def allows_shape(self, shape: tuple[int, ...]) -> bool:
        if isinstance(self.value, Number):
            return True
        assert isinstance(self.value, np.ndarray)
        try:
            return np.broadcast_shapes(self.value.shape, shape) == shape
        except ValueError:
            return False


class UniformInitializer(ElementwiseInitializer):
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
    def config(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b}


class NormalInitializer(ElementwiseInitializer):
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
    def config(self) -> dict[str, Any]:
        return {"mean": self.mean, "stddev": self.stddev}


class DirichletInitializer(Initializer):
    """A symbolic Dirichlet initializer, which initializes all the entries of a tensor
    along one axis by sampling independently from a Dirichlet distribution."""

    def __init__(self, alpha: float | list[float] = 1.0, *, axis: int = -1) -> None:
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
    def config(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "axis": self.axis}

    def allows_shape(self, shape: tuple[int, ...]) -> bool:
        axis = self.axis + len(shape) if self.axis < 0 else self.axis
        if axis >= len(shape):
            return False
        if isinstance(self.alpha, float):
            return True
        assert isinstance(self.alpha, list)
        return shape[axis] == len(self.alpha)


class MixingWeightInitializer(Initializer):
    def __init__(self, initializer: Initializer, fill_value: float = 0.0):
        """A symbolic initializer that describes the initialization of the weights of a sum
        layer with arity > 1, using the pattern of a "mixing layer", i.e., a sum layer that
        computes an element-wise weighted summation of the input vectors.

        Args:
            initializer: The symbolic initializer used to initialize the mixing weights.
            fill_value: The value used to fill the weight entries outside the mixing pattern.
        """
        self._initializer = initializer
        self._fill_value = fill_value

    @property
    def initializer(self) -> Initializer:
        return self._initializer

    @property
    def fill_value(self) -> float:
        return self._fill_value

    @property
    def config(self) -> dict[str, Any]:
        return {"initializer": self._initializer, "fill_value": self._fill_value}

    def allows_shape(self, shape: tuple[int, ...]) -> bool:
        if len(shape) != 2 or shape[1] % shape[0]:
            return False
        mixing_weights_shape = (shape[0], shape[1] // shape[0])
        return self._initializer.allows_shape(mixing_weights_shape)
