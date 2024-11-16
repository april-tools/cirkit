import functools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cirkit.symbolic.circuit import InputLayerFactory
from cirkit.symbolic.dtypes import DataType
from cirkit.symbolic.initializers import (
    DirichletInitializer,
    Initializer,
    MixingWeightInitializer,
    NormalInitializer,
    UniformInitializer,
)
from cirkit.symbolic.layers import BinomialLayer, CategoricalLayer, EmbeddingLayer, GaussianLayer
from cirkit.symbolic.parameters import (
    ClampParameter,
    Parameter,
    ParameterFactory,
    SoftmaxParameter,
    TensorParameter,
    UnaryParameterOp,
)
from cirkit.templates.region_graph import (
    PoonDomingos,
    QuadGraph,
    QuadTree,
    RandomBinaryTree,
    RegionGraph,
)
from cirkit.utils.scope import Scope


@dataclass(frozen=True)
class Parameterization:
    """The settings for a parameterization: the initialization method, the activation
    function to use, and the data type of the parameter tensor."""

    initialization: str
    """The initialization method."""
    activation: str = "none"
    """The activation function. Defaults to 'none', i.e., no activation."""
    dtype: str = "real"
    """The data type. Defaults to 'real', i.e., real numbers."""


def build_image_region_graph(
    name: str,
    image_shape: tuple[int, int],
) -> RegionGraph:
    """Build a region graph that is tailored for image data.

    Args:
        name: The name of the region graph. It can be one of the following: 'quad-tree-2',
            'quad-tree-4', 'quad-graph', 'random-binary-tree', 'poon-domingos'.
            For the Poon-Domingos region graph, the delta parameter used to split patches is
            automatically set to max(ceil(H/8), ceil(W/8)) for images of shape (H, W).
        image_shape: The shape of the image.

    Returns:
        RegionGraph: A region graph.

    Raises:
        ValueError: If the given region graph name is not known.
    """
    match name:
        case "quad-tree-2":
            return QuadTree(image_shape, num_patch_splits=2)
        case "quad-tree-4":
            return QuadTree(image_shape, num_patch_splits=4)
        case "quad-graph":
            return QuadGraph(image_shape)
        case "random-binary-tree":
            return RandomBinaryTree(np.prod(image_shape))
        case "poon-domingos":
            delta = max(np.ceil(image_shape[0] / 8), np.ceil(image_shape[1] / 8))
            return PoonDomingos(image_shape, delta=delta)
        case _:
            raise ValueError(f"Unknown region graph called {name}")


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
            return functools.partial(_embedding_layer_factory, **kwargs)
        case "categorical":
            return functools.partial(_categorical_layer_factory, **kwargs)
        case "binomial":
            return functools.partial(_binomial_layer_factory, **kwargs)
        case "gaussian":
            return functools.partial(_gaussian_layer_factory, **kwargs)
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
    unary_op_factory = name_to_parameter_activation(param.activation)
    dtype = name_to_dtype(param.dtype)
    initializer = name_to_initializer(param.initialization)
    return functools.partial(
        _build_tensor_parameter,
        unary_op_factory=unary_op_factory,
        dtype=dtype,
        initializer=initializer,
    )


def convex_nary_sum_parameterization_factory(shape: tuple[int, ...]) -> Parameter:
    """Construct the parameter of a sum layer with arity > 1 what encodes a convex combination
    of the input vectors to it.

    Args:
        shape: The shape of the parameter. It must be (num_units, arity * num_units), where
            num_units is the size of the input vectors, and arity is the number of them.

    Returns:
        Parameter: A symbolic parameter.

    Raises:
        ValueError: If the given shape is not valid as per its description.
    """
    if len(shape) != 2 or shape[1] % shape[0]:
        raise ValueError(f"Expected shape (num_units, arity * num_units), but found {shape}")
    num_units = shape[0]
    arity = shape[1] // num_units
    shape = (num_units, num_units * arity)
    return Parameter.from_unary(
        SoftmaxParameter(shape),
        TensorParameter(
            *shape,
            learnable=True,
            initializer=MixingWeightInitializer(NormalInitializer(), fill_value=-float("inf")),
        ),
    )


def name_to_parameter_activation(
    name: str, **kwargs
) -> Callable[[tuple[int, ...]], UnaryParameterOp] | None:
    """Retrieves a symbolic unary parameter operator by name.

    Args:
        name: The name of the parameter activation. It can be either 'none',
            'softmax', or 'positive-clamp'.
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
    match name:
        case "uniform":
            return UniformInitializer(0.0, 1.0)
        case "normal":
            return NormalInitializer(0.0, 1.0)
        case "dirichlet":
            return DirichletInitializer(1.0, **kwargs)
        case _:
            raise ValueError(f"Unknown initializer called {name}")


def _build_tensor_parameter(
    shape: tuple[int, ...],
    *,
    unary_op_factory: Callable[[tuple[int, ...]], UnaryParameterOp] | None,
    dtype: DataType,
    initializer: Initializer,
) -> Parameter:
    tensor = TensorParameter(*shape, dtype=dtype, initializer=initializer)
    if unary_op_factory is None:
        return Parameter.from_input(tensor)
    return Parameter.from_unary(unary_op_factory(shape), tensor)


def _embedding_layer_factory(
    scope: Scope,
    num_units: int,
    num_channels: int,
    *,
    num_states: int,
    weight_factory: ParameterFactory | None = None,
) -> EmbeddingLayer:
    return EmbeddingLayer(
        scope, num_units, num_channels, num_states=num_states, weight_factory=weight_factory
    )


def _categorical_layer_factory(
    scope: Scope,
    num_units: int,
    num_channels: int,
    *,
    num_categories: int,
    probs_factory: ParameterFactory | None = None,
    logits_factory: ParameterFactory | None = None,
) -> CategoricalLayer:
    return CategoricalLayer(
        scope,
        num_units,
        num_channels,
        num_categories=num_categories,
        probs_factory=probs_factory,
        logits_factory=logits_factory,
    )


def _binomial_layer_factory(
    scope: Scope,
    num_units: int,
    num_channels: int,
    *,
    total_count: int,
    probs_factory: ParameterFactory | None = None,
    logits_factory: ParameterFactory | None = None,
) -> BinomialLayer:
    return BinomialLayer(
        scope,
        num_units,
        num_channels,
        total_count=total_count,
        probs_factory=probs_factory,
        logits_factory=logits_factory,
    )


def _gaussian_layer_factory(
    scope: Scope,
    num_units: int,
    num_channels: int,
    *,
    mean_factory: ParameterFactory | None = None,
    stddev_factory: ParameterFactory | None = None,
) -> GaussianLayer:
    return GaussianLayer(
        scope, num_units, num_channels, mean_factory=mean_factory, stddev_factory=stddev_factory
    )
