import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from cirkit.symbolic.dtypes import DataType
from cirkit.symbolic.initializers import (
    DirichletInitializer,
    Initializer,
    NormalInitializer,
    UniformInitializer,
)
from cirkit.symbolic.layers import (
    BinomialLayer,
    CategoricalLayer,
    EmbeddingLayer,
    GaussianLayer,
    InputLayer,
    ProductLayer,
    SumLayer,
)
from cirkit.symbolic.parameters import (
    ClampParameter,
    Parameter,
    ParameterFactory,
    SigmoidParameter,
    SoftmaxParameter,
    TensorParameter,
    UnaryParameterOp,
)
from cirkit.utils.scope import Scope


@dataclass(frozen=True)
class Parameterization:
    """The settings for a parameterization: the initialization method, the activation
    function to use, and the data type of the parameter tensor."""

    initialization: str = "normal"
    """The initialization method. Defaults to 'normal', i.e., sampling from a standard Gaussian."""
    activation: str = "none"
    """The activation function. Defaults to 'none', i.e., no activation."""
    dtype: str = "real"
    """The data type. Defaults to 'real', i.e., real numbers."""
    initialization_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional arguments to pass to the initializatiot method."""
    activation_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional arguments to pass to the activation function."""


class InputLayerFactory(Protocol):  # pylint: disable=too-few-public-methods
    """The protocol of a factory that constructs input layers."""

    def __call__(self, scope: Scope, num_units: int) -> InputLayer:
        """Constructs an input layer.

        Args:
            scope: The scope of the layer.
            num_units: The number of input units composing the layer.

        Returns:
            InputLayer: An input layer.
        """


class SumLayerFactory(Protocol):  # pylint: disable=too-few-public-methods
    """The protocol of a factory that constructs sum layers."""

    def __call__(self, num_input_units: int, num_output_units: int) -> SumLayer:
        """Constructs a sum layer.

        Args:
            num_input_units: The number of units in each layer that is an input.
            num_output_units: The number of sum units in the layer.

        Returns:
            SumLayer: A sum layer.
        """


class ProductLayerFactory(Protocol):  # pylint: disable=too-few-public-methods
    """The protocol of a factory that constructs product layers."""

    def __call__(self, num_input_units: int, arity: int) -> ProductLayer:
        """Constructs a product layer.

        Args:
            num_input_units: The number of units in each layer that is an input.
            arity: The number of input layers.

        Returns:
            ProductLayer: A product layer.
        """


def name_to_input_layer_factory(name: str, **kwargs) -> InputLayerFactory:
    """Retrieves a factory that constructs symbolic input layers.

    Args:
        name: The name of the input layer. It can be one of the following:
            'embedding', 'categorical', 'gaussian', 'binomial'.
        **kwargs: Arguments to pass to the factory.

    Returns:
        InputLayerFactory: A symbolic input layer factory.

    Raises:
        ValueError: If the given input layer name is not known.
    """
    match name:
        case "embedding":
            return functools.partial(EmbeddingLayer, **kwargs)
        case "categorical":
            return functools.partial(CategoricalLayer, **kwargs)
        case "binomial":
            return functools.partial(BinomialLayer, **kwargs)
        case "gaussian":
            return functools.partial(GaussianLayer, **kwargs)
        case _:
            raise ValueError(f"Unknown input layer called {name}")


def parameterization_to_factory(param: Parameterization) -> ParameterFactory:
    """Given the settings of a parameterization, retrieves a factory that constructs
        symbolic parameters with that parameterization.

    Args:
        param: The parameterization.

    Returns:
        ParameterFactory: A symbolic parameter factory.

    Raises:
        ValueError: If one of the settings in the given parameterization is unknown.
    """
    unary_op_factory = name_to_parameter_activation(param.activation, **param.activation_kwargs)
    dtype = name_to_dtype(param.dtype)
    initializer = name_to_initializer(param.initialization, **param.initialization_kwargs)
    return functools.partial(
        _build_tensor_parameter,
        unary_op_factory=unary_op_factory,
        dtype=dtype,
        initializer=initializer,
    )


def name_to_parameter_activation(
    name: str, **kwargs
) -> Callable[[tuple[int, ...]], UnaryParameterOp] | None:
    """Retrieves a symbolic unary parameter operator by name.

    Args:
        name: The name of the parameter activation. It can be either 'none',
            'softmax', 'sigmoid', or 'positive-clamp'.
        **kwargs: Optional arguments to pass to symbolic unary parameter.

    Returns:
        None: If name is 'none'
        Callable[[tuple[int, ...]], UnaryParameterOp]: If name is not 'none', then a function that
            takes the parameter shape and returns a unary parameter operator is given.

    Raises:
        ValueError: If the given activation name is not known.
    """
    match name:
        case "none":
            return None
        case "softmax":
            return functools.partial(SoftmaxParameter, **kwargs)
        case "sigmoid":
            return functools.partial(SigmoidParameter)
        case "positive-clamp":
            if "vmin" not in kwargs:
                kwargs["vmin"] = 1e-18
            return functools.partial(ClampParameter, **kwargs)
        case _:
            raise ValueError


def name_to_dtype(name: str) -> DataType:
    """Retrieves a data type by name.

    Args:
        name: The name of the data type. It can be either 'integer, 'real' or 'complex'.

    Returns:
        DataType: The corresponding symbolic data type.

    Raises:
        ValueError: If the given data type name is not known.
    """
    match name:
        case "integer":
            return DataType.INTEGER
        case "real":
            return DataType.REAL
        case "complex":
            return DataType.COMPLEX
        case _:
            raise ValueError(f"Unknown data type called {name}")


def name_to_initializer(name: str, **kwargs) -> Initializer:
    """Retrieves a symbolic initializer object by name.

    Args:
        name: The initialization name. It can be one of the following: 'uniform' (in the range 0-1),
            'normal' (with mean 0 and standard deviation 1), 'dirichlet' (with concentration
            parameters 1).
        **kwargs: Optional arguments to pass to the initializer.

    Returns:
        Initializer: The symbolic initializer.

    Raises:
        ValueError: If the given initialization name is not known.
    """
    kwargs = kwargs.copy()
    match name:
        case "uniform":
            if "a" not in kwargs:
                kwargs["a"] = 0.0
            if "b" not in kwargs:
                kwargs["b"] = 1.0
            return UniformInitializer(**kwargs)
        case "normal":
            if "mean" not in kwargs:
                kwargs["mean"] = 0.0
            if "stddev" not in kwargs:
                kwargs["stddev"] = 1.0
            return NormalInitializer(**kwargs)
        case "dirichlet":
            if "alpha" not in kwargs:
                kwargs["alpha"] = 1.0
            return DirichletInitializer(**kwargs)
        case _:
            raise ValueError(f"Unknown initializer called {name}")


def _build_tensor_parameter(
    shape: tuple[int, ...],
    unary_op_factory: Callable[[tuple[int, ...]], UnaryParameterOp] | None,
    dtype: DataType,
    initializer: Initializer,
) -> Parameter:
    tensor = TensorParameter(*shape, dtype=dtype, initializer=initializer)
    if unary_op_factory is None:
        return Parameter.from_input(tensor)
    return Parameter.from_unary(unary_op_factory(shape), tensor)
