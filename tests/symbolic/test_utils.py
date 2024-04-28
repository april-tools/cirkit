from cirkit.symbolic.layers import CategoricalLayer, DenseLayer, MixingLayer, KroneckerLayer, HadamardLayer
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
