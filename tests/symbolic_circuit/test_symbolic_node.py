import pytest

from cirkit.symbolic_circuit.symbolic_node import SymbolicNode, SymbolicSumNode, SymbolicProductNode, SymbolicInputNode

from cirkit.reparams.leaf import ReparamExp
from cirkit.layers.sum_product import TuckerLayer, CPLayer
from cirkit.layers.input.exp_family import ExpFamilyLayer, NormalLayer, CategoricalLayer


def test_symbolic_node() -> None:
    scope = [1, 2]
    node = SymbolicNode(scope)
    assert repr(node) == "SymbolicNode:\nScope: {1, 2}"
    
    with pytest.raises(AssertionError, match="The scope of a node must be non-empty"):
        SymbolicNode([])

def test_symbolic_sum_node() -> None:
    scope = [1, 2]
    num_input_units = 2
    num_output_units = 3
    node = SymbolicSumNode(scope, num_output_units, TuckerLayer, {})
    node.set_placeholder_params(num_input_units, num_output_units, ReparamExp)
    assert "SymbolicSumNode" in repr(node)
    assert "Layer Class: TuckerLayer" in repr(node)
    assert "Output Units: 3" in repr(node)
    assert "Parameter Shape: (1, 2, 2, 3)" in repr(node)

def test_symbolic_product_node() -> None:
    scope = [1, 2]
    num_input_units = 2
    node = SymbolicProductNode(scope, num_input_units)
    assert "SymbolicProductNode" in repr(node)
    assert "Product Class: 'Kroneker Product'" in repr(node)
    assert "Output Units: 4" in repr(node)  # 2 ** 2

def test_symbolic_input_node() -> None:
    scope = [1, 2]
    num_output_units = 3
    efamily_kwargs = {'num_categories': 5}
    node = SymbolicInputNode(scope, num_output_units, CategoricalLayer, efamily_kwargs)
    node.set_placeholder_params(1, 1, ReparamExp)
    assert "SymbolicInputNode" in repr(node)
    assert "Input Exp Family Class: CategoricalLayer" in repr(node)
    assert "Layer KWArgs: {'num_categories': 5}" in repr(node)
    assert "Output Units: 3" in repr(node)
    assert "Parameter Shape: (1, 3, 1, 5)" in repr(node)

