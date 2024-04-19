import pytest

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import CategoricalLayer, DenseLayer, MixingLayer, KroneckerLayer
from cirkit.symbolic.params import SoftmaxParameter
from cirkit.templates.region_graph import QuadTree
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


def kronecker_layer_factory(scope: Scope, num_input_units: int, arity: int) -> KroneckerLayer:
    return KroneckerLayer(scope, num_input_units, arity)


@pytest.mark.parametrize("struct_decomp", [False, True])
def test_build_from_region_graph(struct_decomp: bool):
    qt = QuadTree(shape=(9, 9), struct_decomp=struct_decomp)
    sc = Circuit.from_region_graph(
        qt, num_input_units=4, num_sum_units=8,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory,
        mixing_factory=mixing_layer_factory
    )
    assert sc.is_smooth
    assert sc.is_decomposable
    #if struct_decomp:
        #assert sc.is_structured_decomposable
    #else:
    # assert not sc.is_structured_decomposable
