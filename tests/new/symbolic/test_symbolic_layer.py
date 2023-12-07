# pylint: disable=missing-function-docstring

from cirkit.layers.input.exp_family import CategoricalLayer
from cirkit.layers.sum_product import BaseCPLayer, TuckerLayer
from cirkit.new.reparams import ExpReparam
from cirkit.new.symbolic.symbolic_layer import (
    SymbolicInputLayer,
    SymbolicProductLayer,
    SymbolicSumLayer,
)


def test_symbolic_sum_layer() -> None:
    scope = [1, 2]
    num_units = 3
    layer = SymbolicSumLayer(scope, num_units, TuckerLayer, reparam=ExpReparam())
    assert "SymbolicSumLayer" in repr(layer)
    assert "Scope: frozenset({1, 2})" in repr(layer)
    assert "Layer Class: TuckerLayer" in repr(layer)
    assert "Number of Units: 3" in repr(layer)


def test_symbolic_sum_layer_cp() -> None:
    scope = [1, 2]
    num_units = 3
    layer_kwargs = {"collapsed": False, "shared": False, "arity": 2}
    layer = SymbolicSumLayer(scope, num_units, BaseCPLayer, layer_kwargs, reparam=ExpReparam())
    assert "SymbolicSumLayer" in repr(layer)
    assert "Scope: frozenset({1, 2})" in repr(layer)
    assert "Layer Class: UncollapsedCPLayer" in repr(layer)
    assert "Number of Units: 3" in repr(layer)


def test_symbolic_product_node() -> None:
    scope = [1, 2]
    num_input_units = 2
    layer = SymbolicProductLayer(scope, num_input_units, TuckerLayer)
    assert "SymbolicProductLayer" in repr(layer)
    assert "Scope: frozenset({1, 2})" in repr(layer)
    assert "Layer Class: TuckerLayer" in repr(layer)
    assert "Number of Units: 2" in repr(layer)


def test_symbolic_input_node() -> None:
    scope = [1, 2]
    num_units = 3
    efamily_kwargs = {"num_categories": 5}
    layer = SymbolicInputLayer(
        scope, num_units, CategoricalLayer, efamily_kwargs, reparam=ExpReparam()
    )
    assert "SymbolicInputLayer" in repr(layer)
    assert "Scope: frozenset({1, 2})" in repr(layer)
    assert "Input Exp Family Class: CategoricalLayer" in repr(layer)
    assert "Layer KWArgs: {'num_categories': 5}" in repr(layer)
    assert "Number of Units: 3" in repr(layer)
