import numpy as np

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import CategoricalLayer, DenseLayer, MixingLayer, KroneckerLayer, HadamardLayer
from cirkit.templates.region_graph import RandomBinaryTree
from cirkit.utils.scope import Scope


def categorical_layer_factory(
        scope: Scope,
        num_units: int,
        num_channels: int
) -> CategoricalLayer:
    return CategoricalLayer(scope, num_units, num_channels, num_categories=256)


def dense_layer_factory(scope: Scope, num_input_units: int, num_output_units: int) -> DenseLayer:
    return DenseLayer(scope, num_input_units, num_output_units)


def mixing_layer_factory(scope: Scope, num_units: int, arity: int) -> MixingLayer:
    return MixingLayer(scope, num_units, arity)


def hadamard_layer_factory(scope: Scope, num_input_units: int, arity: int) -> HadamardLayer:
    return HadamardLayer(scope, num_input_units, arity)


def kronecker_layer_factory(scope: Scope, num_input_units: int, arity: int) -> KroneckerLayer:
    return KroneckerLayer(scope, num_input_units, arity)


def build_circuit(
        num_variables: int,
        num_input_units: int,
        num_sum_units: int,
        num_repetitions: int = 1,
        seed: int = 42
):
    rg = RandomBinaryTree(
        num_variables,
        depth=int(np.floor(np.log2(num_variables))),
        num_repetitions=num_repetitions,
        seed=seed)
    return Circuit.from_region_graph(
        rg, num_input_units=num_input_units, num_sum_units=num_sum_units,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=hadamard_layer_factory,
        mixing_factory=mixing_layer_factory
    )
