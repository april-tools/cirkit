import functools
from typing import Optional

import numpy as np

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import NormalInitializer
from cirkit.symbolic.layers import (
    CategoricalLayer,
    DenseLayer,
    GaussianLayer,
    HadamardLayer,
    KroneckerLayer,
    MixingLayer,
)
from cirkit.symbolic.parameters import (
    ExpParameter,
    LogSoftmaxParameter,
    Parameter,
    ParameterFactory,
    SoftmaxParameter,
    TensorParameter,
)
from cirkit.templates.region_graph import RandomBinaryTree
from cirkit.utils.scope import Scope


def categorical_layer_factory(
    scope: Scope,
    num_units: int,
    num_channels: int,
    *,
    num_categories: int = 2,
    logits_factory: Optional[ParameterFactory] = None,
) -> CategoricalLayer:
    return CategoricalLayer(
        scope, num_units, num_channels, num_categories=num_categories, logits_factory=logits_factory
    )


def gaussian_layer_factory(
    scope: Scope,
    num_units: int,
    num_channels: int,
) -> GaussianLayer:
    return GaussianLayer(scope, num_units, num_channels)


def dense_layer_factory(
    scope: Scope,
    num_input_units: int,
    num_output_units: int,
    *,
    sum_weight_factory: Optional[ParameterFactory] = None,
) -> DenseLayer:
    return DenseLayer(scope, num_input_units, num_output_units, weight_factory=sum_weight_factory)


def mixing_layer_factory(
    scope: Scope,
    num_units: int,
    arity: int,
    *,
    mixing_weight_factory: Optional[ParameterFactory] = None,
) -> MixingLayer:
    return MixingLayer(scope, num_units, arity, weight_factory=mixing_weight_factory)


def hadamard_layer_factory(scope: Scope, num_input_units: int, arity: int) -> HadamardLayer:
    return HadamardLayer(scope, num_input_units, arity)


def kronecker_layer_factory(scope: Scope, num_input_units: int, arity: int) -> KroneckerLayer:
    return KroneckerLayer(scope, num_input_units, arity)


def build_simple_circuit(
    num_variables: int,
    num_input_units: int,
    num_sum_units: int,
    num_repetitions: int = 1,
    seed: int = 42,
    sum_product_layer: str = "cp",
    input_layer: str = "categorical",
    sum_weight_factory: Optional[ParameterFactory] = None,
    logits_factory: Optional[ParameterFactory] = None,
):
    rg = RandomBinaryTree(
        num_variables,
        depth=int(np.floor(np.log2(num_variables))),
        num_repetitions=num_repetitions,
        seed=seed,
    )
    if sum_product_layer == "cp":
        prod_factory = hadamard_layer_factory
    elif sum_product_layer == "tucker":
        prod_factory = kronecker_layer_factory
    else:
        assert False
    if input_layer == "categorical":
        input_factory = functools.partial(categorical_layer_factory, logits_factory=logits_factory)
    elif input_layer == "gaussian":
        input_factory = gaussian_layer_factory
    else:
        assert False
    return Circuit.from_region_graph(
        rg,
        num_input_units=num_input_units,
        num_sum_units=num_sum_units,
        input_factory=input_factory,
        sum_factory=functools.partial(dense_layer_factory, sum_weight_factory=sum_weight_factory),
        prod_factory=prod_factory,
        mixing_factory=functools.partial(
            mixing_layer_factory, mixing_weight_factory=sum_weight_factory
        ),
    )


def build_simple_pc(
    num_variables: int,
    num_input_units: int = 2,
    num_sum_units: int = 2,
    num_repetitions: int = 1,
    seed: int = 42,
    sum_product_layer: str = "cp",
    input_layer: str = "categorical",
    normalized: bool = False,
):
    if normalized:
        sum_weight_factory = lambda shape: Parameter.from_unary(
            SoftmaxParameter(shape, axis=1),
            TensorParameter(*shape, initializer=NormalInitializer(0.0, 3e-1)),
        )
    else:
        sum_weight_factory = lambda shape: Parameter.from_unary(
            ExpParameter(shape), TensorParameter(*shape, initializer=NormalInitializer(0.0, 3e-1))
        )
    if normalized:
        logits_factory = lambda shape: Parameter.from_unary(
            LogSoftmaxParameter(shape, axis=3),
            TensorParameter(*shape, initializer=NormalInitializer(0.0, 3e-1)),
        )
    else:
        logits_factory = lambda shape: Parameter.from_leaf(
            TensorParameter(*shape, initializer=NormalInitializer(0.0, 3e-1))
        )
    return build_simple_circuit(
        num_variables,
        num_input_units,
        num_sum_units,
        num_repetitions=num_repetitions,
        seed=seed,
        sum_product_layer=sum_product_layer,
        input_layer=input_layer,
        sum_weight_factory=sum_weight_factory,
        logits_factory=logits_factory,
    )
