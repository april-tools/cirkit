from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import DirichletInitializer
from cirkit.symbolic.layers import CategoricalLayer, SumLayer
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
        probs=Parameter.from_input(
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
    assert all(
        isinstance(sl, CategoricalLayer) and len(sc.layer_scope(sl)) == 1 for sl in sc.inputs
    )
    assert len(list(sc.product_layers)) == 14
    assert len(list(sl for sl in sc.sum_layers if isinstance(sl, SumLayer) and sl.arity == 1)) == 30
    assert len(list(sl for sl in sc.sum_layers if isinstance(sl, SumLayer) and sl.arity > 1)) == 2
    assert (
        len(list(sl for sl in sc.product_layers if sc.layer_scope(sl) == Scope([0, 1, 3, 4]))) == 2
    )
    assert len(list(sl for sl in sc.product_layers if sc.layer_scope(sl) == Scope(range(9)))) == 2
    (out_sl,) = sc.outputs
    assert isinstance(out_sl, SumLayer) and out_sl.arity > 1


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
    assert all(
        isinstance(sl, CategoricalLayer) and len(sc.layer_scope(sl)) == 1 for sl in sc.inputs
    )
    assert len(list(sc.product_layers)) == 4
    assert len(list(sl for sl in sc.sum_layers if isinstance(sl, SumLayer) and sl.arity == 1)) == 13
    assert len(list(sl for sl in sc.sum_layers if isinstance(sl, SumLayer) and sl.arity > 1)) == 0
    assert all(
        len(sc.layer_inputs(sl)) == 2 for sl in sc.product_layers if len(sc.layer_scope(sl)) == 2
    )
    assert all(
        len(sc.layer_inputs(sl)) == 4 for sl in sc.product_layers if len(sc.layer_scope(sl)) > 2
    )
    (out_sl,) = sc.outputs
    assert isinstance(out_sl, SumLayer) and out_sl.arity == 1
