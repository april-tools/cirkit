import functools
from typing import Optional

import numpy as np

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import Initializer, NormalInitializer
from cirkit.symbolic.layers import (
    CategoricalLayer,
    DenseLayer,
    HadamardLayer,
    KroneckerLayer,
    MixingLayer,
)
from cirkit.symbolic.parameters import (
    ExpParameter,
    LogSoftmaxParameter,
    Parameter,
    Parameterization,
    SoftmaxParameter,
)
from cirkit.templates.region_graph import RandomBinaryTree
from cirkit.utils.scope import Scope


def categorical_layer_factory(
    scope: Scope,
    num_units: int,
    num_channels: int,
    num_categories: int = 2,
    logits_parameterization: Optional[Parameterization] = None,
    logits_initializer: Optional[Initializer] = None,
) -> CategoricalLayer:
    return CategoricalLayer(
        scope,
        num_units,
        num_channels,
        num_categories=num_categories,
        logits_parameterization=logits_parameterization,
        logits_initializer=logits_initializer,
    )


def dense_layer_factory(
    scope: Scope,
    num_input_units: int,
    num_output_units: int,
    parameterization: Optional[Parameterization] = None,
    initializer: Optional[Initializer] = None,
) -> DenseLayer:
    return DenseLayer(
        scope,
        num_input_units,
        num_output_units,
        parameterization=parameterization,
        initializer=initializer,
    )


def mixing_layer_factory(
    scope: Scope,
    num_units: int,
    arity: int,
    parameterization: Optional[Parameterization] = None,
    initializer: Optional[Initializer] = None,
) -> MixingLayer:
    return MixingLayer(
        scope, num_units, arity, parameterization=parameterization, initializer=initializer
    )


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
    input_layer: str = "categorical",
    sum_parameterization: Optional[Parameterization] = None,
    logits_parameterization: Optional[Parameterization] = None,
    sum_initializer: Optional[Initializer] = None,
    logits_initializer: Optional[Initializer] = None,
):
    rg = RandomBinaryTree(
        num_variables,
        depth=int(np.floor(np.log2(num_variables))),
        num_repetitions=num_repetitions,
        seed=seed,
    )
    if input_layer == "categorical":
        input_layer = functools.partial(
            categorical_layer_factory,
            logits_parameterization=logits_parameterization,
            logits_initializer=logits_initializer,
        )
    else:
        assert False
    return Circuit.from_region_graph(
        rg,
        num_input_units=num_input_units,
        num_sum_units=num_sum_units,
        input_factory=input_layer,
        sum_factory=functools.partial(
            dense_layer_factory, parameterization=sum_parameterization, initializer=sum_initializer
        ),
        prod_factory=hadamard_layer_factory,
        mixing_factory=functools.partial(
            mixing_layer_factory, parameterization=sum_parameterization, initializer=sum_initializer
        ),
    )


def build_simple_pc(
    num_variables: int,
    num_input_units: int = 2,
    num_sum_units: int = 2,
    num_repetitions: int = 1,
    seed: int = 42,
    input_layer: str = "categorical",
    normalized: bool = False,
):
    if normalized:
        sum_param = lambda p: Parameter.from_unary(p, SoftmaxParameter(p.shape, axis=1))
    else:
        sum_param = lambda p: Parameter.from_unary(p, ExpParameter(p.shape))
    if normalized:
        logits_param = lambda p: Parameter.from_unary(p, LogSoftmaxParameter(p.shape, axis=3))
    else:
        logits_param = None
    return build_simple_circuit(
        num_variables,
        num_input_units,
        num_sum_units,
        num_repetitions=num_repetitions,
        seed=seed,
        input_layer=input_layer,
        sum_parameterization=sum_param,
        logits_parameterization=logits_param,
        sum_initializer=NormalInitializer(0.0, 3e-1),
        logits_initializer=NormalInitializer(0.0, 3e-1),
    )
