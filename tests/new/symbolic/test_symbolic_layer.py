# type: ignore
# pylint: skip-file

import pytest

from cirkit.layers.input.exp_family import CategoricalLayer
from cirkit.layers.sum_product import BaseCPLayer, TuckerLayer
from cirkit.new.symbolic.symbolic_layer import (
    SymbolicInputLayer,
    SymbolicLayer,
    SymbolicProductLayer,
    SymbolicSumLayer,
)
from cirkit.reparams.leaf import ReparamExp


def test_symbolic_node() -> None:
    scope = [1, 2]
    node = SymbolicLayer(scope)
    assert repr(node) == "SymbolicLayer:\nScope: {1, 2}\n"

    with pytest.raises(AssertionError, match="The scope of a node must be non-empty"):
        SymbolicLayer([])


def test_symbolic_sum_node() -> None:
    scope = [1, 2]
    num_input_units = 2
    num_units = 3
    node = SymbolicSumLayer(scope, num_units, TuckerLayer, {})
    node.set_placeholder_params(num_input_units, num_units, ReparamExp)
    assert "SymbolicSumLayer" in repr(node)
    assert "Scope: frozenset({1, 2})" in repr(node)
    assert "Layer Class: TuckerLayer" in repr(node)
    assert "Number of Units: 3" in repr(node)
    assert "Parameter Shape: (1, 2, 2, 3)" in repr(node)
    assert "CP Layer Parameter in Shape: None" in repr(node)
    assert "CP Layer Parameter out Shape: None" in repr(node)


def test_symbolic_sum_node_cp() -> None:
    scope = [1, 2]
    num_input_units = 2
    num_units = 3
    layer_kwargs = {"collapsed": False, "shared": False, "arity": 2}
    node = SymbolicSumLayer(scope, num_units, BaseCPLayer, layer_kwargs)
    node.set_placeholder_params(num_input_units, num_units, ReparamExp)
    assert "SymbolicSumLayer" in repr(node)
    assert "Scope: frozenset({1, 2})" in repr(node)
    assert "Layer Class: UncollapsedCPLayer" in repr(node)
    assert "Number of Units: 3" in repr(node)
    assert "Parameter Shape: None" in repr(node)
    assert "CP Layer Parameter in Shape: (1, 2, 2, 1)" in repr(node)
    assert "CP Layer Parameter out Shape: (1, 1, 3)" in repr(node)


def test_symbolic_product_node() -> None:
    scope = [1, 2]
    num_input_units = 2
    node = SymbolicProductLayer(scope, num_input_units, TuckerLayer)
    assert "SymbolicProductLayer" in repr(node)
    assert "Scope: frozenset({1, 2})" in repr(node)
    assert "Layer Class: TuckerLayer" in repr(node)
    assert "Number of Units: 2" in repr(node)


def test_symbolic_input_node() -> None:
    scope = [1, 2]
    num_units = 3
    efamily_kwargs = {"num_categories": 5}
    node = SymbolicInputLayer(scope, num_units, CategoricalLayer, efamily_kwargs)
    node.set_placeholder_params(1, 1, ReparamExp)
    assert "SymbolicInputLayer" in repr(node)
    assert "Scope: frozenset({1, 2})" in repr(node)
    assert "Input Exp Family Class: CategoricalLayer" in repr(node)
    assert "Layer KWArgs: {'num_categories': 5}" in repr(node)
    assert "Number of Units: 3" in repr(node)
    assert "Parameter Shape: (1, 3, 1, 5)" in repr(node)
