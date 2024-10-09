from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import DirichletInitializer
from cirkit.symbolic.layers import CategoricalLayer, DenseLayer, MixingLayer
from cirkit.symbolic.parameters import Parameter, TensorParameter
from cirkit.templates.region_graph import QuadGraph, QuadTree
from cirkit.utils.scope import Scope


def categorical_layer_factory(
    scope: Scope, num_units: int, num_channels: int, *, num_categories: int = 2
) -> CategoricalLayer:
    return CategoricalLayer(
        scope,
        num_units,
        num_channels,
        num_categories=num_categories,
        probs=Parameter.from_leaf(
            TensorParameter(
                num_units, num_channels, num_categories, initializer=DirichletInitializer()
            )
        ),
    )


def test_build_circuit_qg_3x3_cp():
    rg = QuadGraph((3, 3))
    sc = Circuit.from_region_graph(
        rg,
        num_input_units=3,
        num_sum_units=2,
        sum_product="cp",
        input_factory=categorical_layer_factory,
    )
    assert sc.is_smooth
    assert sc.is_decomposable
    assert not sc.is_structured_decomposable
    assert not sc.is_omni_compatible
    assert len(list(sc.inputs)) == 9
    assert all(isinstance(sl, CategoricalLayer) and len(sl.scope) == 1 for sl in sc.inputs)
    assert len(list(sc.product_layers)) == 14
    assert len(list(sl for sl in sc.sum_layers if isinstance(sl, DenseLayer))) == 28
    assert len(list(sl for sl in sc.sum_layers if isinstance(sl, MixingLayer))) == 2
    assert len(list(sl for sl in sc.product_layers if sl.scope == Scope([0, 1, 3, 4]))) == 2
    assert len(list(sl for sl in sc.product_layers if sl.scope == Scope(range(9)))) == 2


def test_build_circuit_qt4_3x3_cp():
    rg = QuadTree((3, 3), num_patch_splits=4)
    sc = Circuit.from_region_graph(
        rg,
        num_input_units=3,
        num_sum_units=2,
        sum_product="cp",
        input_factory=categorical_layer_factory,
    )
    assert sc.is_smooth
    assert sc.is_decomposable
    assert sc.is_structured_decomposable
    assert len(list(sc.inputs)) == 9
    assert all(isinstance(sl, CategoricalLayer) and len(sl.scope) == 1 for sl in sc.inputs)
    assert len(list(sc.product_layers)) == 4
    assert len(list(sl for sl in sc.sum_layers if isinstance(sl, DenseLayer))) == 12
    assert len(list(sl for sl in sc.sum_layers if isinstance(sl, MixingLayer))) == 0
    assert all(len(sc.layer_inputs(sl)) == 2 for sl in sc.product_layers if len(sl.scope) == 2)
    assert all(len(sc.layer_inputs(sl)) == 4 for sl in sc.product_layers if len(sl.scope) > 2)
