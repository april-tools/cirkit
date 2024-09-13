import functools
from typing import Optional, Tuple

import numpy as np

from cirkit.symbolic.circuit import InputLayerFactory
from cirkit.symbolic.initializers import (
    DirichletInitializer,
    Initializer,
    NormalInitializer,
    UniformInitializer,
)
from cirkit.symbolic.layers import CategoricalLayer, GaussianLayer, MixingLayer
from cirkit.symbolic.parameters import (
    ClampParameter,
    Parameter,
    ParameterFactory,
    SoftmaxParameter,
    TensorParameter,
)
from cirkit.templates.region_graph import PoonDomingos, QuadGraph, QuadTree, RegionGraph
from cirkit.utils.scope import Scope


def build_image_region_graph(
    name: str,
    image_shape: Tuple[int, int],
) -> RegionGraph:
    """
    Build a region graph that is tailored for image data.

    Args:
        name: The name of the region graph. It can be one of the following: 'quad-tree-2',
         'quad-tree-4', 'quad-graph', 'poon-domingos'. For the Poon-Domingos region graph, the delta
         parameter used to split patches is automatically set to max(ceil(H/8), ceil(W/8)) for
         images of shape (H, W).
        image_shape: The shape of the image.

    Returns:
        RegionGraph: A region graph.
    """
    if name == "quad-tree-2":
        return QuadTree(image_shape, num_patch_splits=2)
    if name == "quad-tree-4":
        return QuadTree(image_shape, num_patch_splits=4)
    if name == "quad-graph":
        return QuadGraph(image_shape)
    if name == "poon-domingos":
        delta = max(np.ceil(image_shape[0] / 8), np.ceil(image_shape[1] / 8))
        return PoonDomingos(image_shape, delta=delta)
    raise ValueError(f"Unknown region graph called {name}")


def name_to_input_layer_factory(name: str, **kwargs) -> InputLayerFactory:
    """
    Retrieves a factory that constructs symbolic input layers.

    Args:
        name: The name of the input layer. It can be one of the following: 'categorical', 'gaussian'.
        **kwargs: Arguments to pass to the factory.

    Returns:
        InputLayerFactory: A symbolic input layer factory.
    """
    if name == "categorical":
        return functools.partial(_categorical_layer_factory, **kwargs)
    if name == "gaussian":
        return functools.partial(_gaussian_layer_factory, **kwargs)
    raise ValueError(f"Unknown input layer called {name}")


def name_to_parameter_factory(name: str, **kwargs) -> ParameterFactory:
    """
    Retrieves a factory that constructs symbolic parameters.

    Args:
        name: The name of the parameterization to use. It can be one of the following:
         'id' (identity), 'softmax', 'positive-clamp' (equivalent to max(., 1e-18)).
        **kwargs: Arguments to pass to the symbolic tensor parameter (TensorParameter).

    Returns:
        ParameterFactory: A symbolic parameter factory.
    """
    if name == "id":
        return functools.partial(_id_parameter_factory, **kwargs)
    if name == "softmax":
        return functools.partial(_softmax_parameter_factory, **kwargs)
    if name == "positive-clamp":
        return functools.partial(_positive_clamp_parameter_factory, **kwargs)
    raise ValueError(f"Unknown parameterization called {name}")


def name_to_initializer(name: str, **kwargs) -> Initializer:
    """
    Retrieves a symbolic initializer object.

    Args:
        name: The initializer name. It can be one of the following: 'uniform' (in the range 0-1),
         'normal' (with mean 0 and standard deviation 1), 'dirichlet' (with concentration
         parameters 1).
        **kwargs: Optional arguments to pass to the initializer.

    Returns:
        Initializer: The symbolic initializer.
    """
    if name == "uniform":
        return UniformInitializer(0.0, 1.0)
    if name == "normal":
        return NormalInitializer(0.0, 1.0)
    if name == "dirichlet":
        return DirichletInitializer(1.0, **kwargs)
    raise ValueError(f"Unknown initializer called {name}")


def mixing_layer_factory(
    scope: Scope, num_units: int, arity: int, *, weight_factory: Optional[ParameterFactory] = None
) -> MixingLayer:
    """
    Build a mixing layer, given hyperparameters and an optional weight factory.

    Args:
        scope: The scope.
        num_units: The number of sum units in the layer.
        arity: The arity, i.e., the number of input layers.
        weight_factory: An optional factory constructing symbolic weights.

    Returns:
        MixingLayer: A mixing layer.
    """
    return MixingLayer(scope, num_units, arity, weight_factory=weight_factory)


def _categorical_layer_factory(
    scope: Scope,
    num_units: int,
    num_channels: int,
    *,
    num_categories: int,
    probs_factory: Optional[ParameterFactory] = None,
    logits_factory: Optional[ParameterFactory] = None,
) -> CategoricalLayer:
    return CategoricalLayer(
        scope,
        num_units,
        num_channels,
        num_categories=num_categories,
        probs_factory=probs_factory,
        logits_factory=logits_factory,
    )


def _gaussian_layer_factory(
    scope: Scope,
    num_units: int,
    num_channels: int,
    *,
    mean_factory: Optional[ParameterFactory] = None,
    stddev_factory: Optional[ParameterFactory] = None,
) -> GaussianLayer:
    return GaussianLayer(
        scope, num_units, num_channels, mean_factory=mean_factory, stddev_factory=stddev_factory
    )


def _id_parameter_factory(shape: Tuple[int, ...], **kwargs) -> Parameter:
    return Parameter.from_leaf(TensorParameter(*shape, **kwargs))


def _softmax_parameter_factory(shape: Tuple[int, ...], *, axis: int = -1, **kwargs) -> Parameter:
    return Parameter.from_unary(
        SoftmaxParameter(shape, axis=axis), TensorParameter(*shape, **kwargs)
    )


def _positive_clamp_parameter_factory(shape: Tuple[int, ...], **kwargs) -> Parameter:
    return Parameter.from_unary(
        ClampParameter(shape, vmin=1e-18), TensorParameter(*shape, **kwargs)
    )
