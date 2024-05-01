import functools
from typing import Optional

import numpy as np

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import CategoricalLayer, DenseLayer, MixingLayer, KroneckerLayer, HadamardLayer, \
    GaussianLayer
from cirkit.symbolic.params import Parameterization, SoftmaxParameter, ExpParameter, LogSoftmaxParameter
from cirkit.templates.region_graph import RandomBinaryTree
from cirkit.utils.scope import Scope


def categorical_layer_factory(
    scope: Scope,
    num_units: int,
    num_channels: int,
    num_categories: int = 2,
    logits_param: Optional[Parameterization] = None
) -> CategoricalLayer:
    return CategoricalLayer(
        scope, num_units, num_channels, num_categories=num_categories, logits_param=logits_param
    )


def gaussian_layer_factory(
    scope: Scope,
    num_units: int,
    num_channels: int
) -> GaussianLayer:
    return GaussianLayer(scope, num_units, num_channels)


def dense_layer_factory(
    scope: Scope,
    num_input_units: int,
    num_output_units: int,
    weight_param: Optional[Parameterization] = None
) -> DenseLayer:
    return DenseLayer(scope, num_input_units, num_output_units, weight_param=weight_param)


def mixing_layer_factory(
    scope: Scope,
    num_units: int,
    arity: int,
    weight_param: Optional[Parameterization] = None
) -> MixingLayer:
    return MixingLayer(scope, num_units, arity, weight_param=weight_param)


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
        input_layer: str = 'categorical',
        sum_param: Optional[Parameterization] = None,
        logits_param: Optional[Parameterization] = None
):
    rg = RandomBinaryTree(
        num_variables,
        depth=int(np.ceil(np.log2(num_variables))),
        num_repetitions=num_repetitions,
        seed=seed)
    if input_layer == 'categorical':
        input_layer = functools.partial(categorical_layer_factory, logits_param=logits_param)
    elif input_layer == 'gaussian':
        input_layer = gaussian_layer_factory
    else:
        assert False
    return Circuit.from_region_graph(
        rg, num_input_units=num_input_units, num_sum_units=num_sum_units,
        input_factory=input_layer,
        sum_factory=functools.partial(dense_layer_factory, weight_param=sum_param),
        prod_factory=hadamard_layer_factory,
        mixing_factory=functools.partial(mixing_layer_factory, weight_param=sum_param)
    )


def build_simple_pc(
        num_variables: int,
        num_input_units: int,
        num_sum_units: int,
        num_repetitions: int = 1,
        seed: int = 42,
        input_layer: str = 'categorical',
        normalized: bool = False
):
    if normalized:
        sum_param = lambda p: SoftmaxParameter(p, axis=-1)
    else:
        sum_param = lambda p: ExpParameter(p)
    if normalized:
        logits_param = lambda p: LogSoftmaxParameter(p, axis=-1)
    else:
        logits_param = None
    return build_simple_circuit(
        num_variables, num_input_units, num_sum_units, num_repetitions=num_repetitions, seed=seed,
        input_layer=input_layer, sum_param=sum_param, logits_param=logits_param
    )
