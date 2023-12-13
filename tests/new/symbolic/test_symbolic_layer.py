# pylint: disable=missing-function-docstring

from typing import Dict

from cirkit.new.layers import CategoricalLayer, DenseLayer, HadamardLayer, TuckerLayer
from cirkit.new.reparams import ExpReparam
from cirkit.new.symbolic import SymbolicInputLayer, SymbolicProductLayer, SymbolicSumLayer
from tests.new.symbolic.test_utils import get_simple_rg

# TODO: avoid repetition?


def test_symbolic_layers_sum_and_prod() -> None:
    rg = get_simple_rg()
    input_node0, input_node1 = rg.input_nodes
    (partition_node,) = rg.partition_nodes
    (region_node,) = rg.inner_region_nodes

    num_units = 3
    input_kwargs = {"num_categories": 5}
    sum_kwargs: Dict[str, None] = {}  # Avoid Any.
    reparam = ExpReparam()

    input_layer0 = SymbolicInputLayer(
        input_node0,
        (),
        num_units=num_units,
        layer_cls=CategoricalLayer,
        layer_kwargs=input_kwargs,
        reparam=reparam,
    )
    assert "SymbolicInputLayer" in repr(input_layer0)
    assert "Scope: Scope({0})" in repr(input_layer0)
    assert "Input Exp Family Class: CategoricalLayer" in repr(input_layer0)
    assert "Layer KWArgs: {'num_categories': 5}" in repr(input_layer0)
    assert "Number of Units: 3" in repr(input_layer0)
    input_layer1 = SymbolicInputLayer(
        input_node1,
        (),
        num_units=num_units,
        layer_cls=CategoricalLayer,
        layer_kwargs=input_kwargs,
        reparam=reparam,
    )

    prod_layer = SymbolicProductLayer(
        partition_node,
        (input_layer0, input_layer1),
        num_units=num_units,
        layer_cls=HadamardLayer,
    )
    assert "SymbolicProductLayer" in repr(prod_layer)
    assert "Scope: Scope({0, 1})" in repr(prod_layer)
    assert "Layer Class: HadamardLayer" in repr(prod_layer)
    assert "Number of Units: 3" in repr(prod_layer)

    sum_layer = SymbolicSumLayer(
        region_node,
        (prod_layer,),
        num_units=num_units,
        layer_cls=DenseLayer,
        layer_kwargs=sum_kwargs,
        reparam=reparam,
    )
    assert "SymbolicSumLayer" in repr(sum_layer)
    assert "Scope: Scope({0, 1})" in repr(sum_layer)
    assert "Layer Class: DenseLayer" in repr(sum_layer)
    assert "Number of Units: 3" in repr(sum_layer)


def test_symbolic_layers_sum_prod() -> None:
    rg = get_simple_rg()
    input_node0, input_node1 = rg.input_nodes
    (partition_node,) = rg.partition_nodes
    (region_node,) = rg.inner_region_nodes

    num_units = 3
    input_kwargs = {"num_categories": 5}
    sum_kwargs: Dict[str, None] = {}  # Avoid Any.
    reparam = ExpReparam()

    input_layer0 = SymbolicInputLayer(
        input_node0,
        (),
        num_units=num_units,
        layer_cls=CategoricalLayer,
        layer_kwargs=input_kwargs,
        reparam=reparam,
    )
    assert "SymbolicInputLayer" in repr(input_layer0)
    assert "Scope: Scope({0})" in repr(input_layer0)
    assert "Input Exp Family Class: CategoricalLayer" in repr(input_layer0)
    assert "Layer KWArgs: {'num_categories': 5}" in repr(input_layer0)
    assert "Number of Units: 3" in repr(input_layer0)
    input_layer1 = SymbolicInputLayer(
        input_node1,
        (),
        num_units=num_units,
        layer_cls=CategoricalLayer,
        layer_kwargs=input_kwargs,
        reparam=reparam,
    )

    prod_layer = SymbolicProductLayer(
        partition_node,
        (input_layer0, input_layer1),
        num_units=num_units**2,
        layer_cls=TuckerLayer,
    )
    assert "SymbolicProductLayer" in repr(prod_layer)
    assert "Scope: Scope({0, 1})" in repr(prod_layer)
    assert "Layer Class: TuckerLayer" in repr(prod_layer)
    assert "Number of Units: 9" in repr(prod_layer)

    sum_layer = SymbolicSumLayer(
        region_node,
        (prod_layer,),
        num_units=num_units,
        layer_cls=TuckerLayer,
        layer_kwargs=sum_kwargs,
        reparam=reparam,
    )
    assert "SymbolicSumLayer" in repr(sum_layer)
    assert "Scope: Scope({0, 1})" in repr(sum_layer)
    assert "Layer Class: TuckerLayer" in repr(sum_layer)
    assert "Number of Units: 3" in repr(sum_layer)
