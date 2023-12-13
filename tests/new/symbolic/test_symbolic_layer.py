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
    assert (
        repr(input_layer0).splitlines()[0] == f"SymbolicInputLayer@0x{id(input_layer0):x}("
        "scope=Scope({0}), num_units=3, layer_cls=<class 'cirkit.new.layers.input.exp_family"
        ".categorical.CategoricalLayer'>, layer_kwargs={'num_categories': 5}, reparam=ExpReparam("
    )

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
    assert (
        repr(prod_layer).splitlines()[0] == f"SymbolicProductLayer@0x{id(prod_layer):x}("
        "scope=Scope({0, 1}), arity=2, num_units=3, layer_cls=<class 'cirkit.new.layers.inner"
        ".product.hadamard.HadamardLayer'>, layer_kwargs={})"
    )

    sum_layer = SymbolicSumLayer(
        region_node,
        (prod_layer,),
        num_units=num_units,
        layer_cls=DenseLayer,
        layer_kwargs=sum_kwargs,
        reparam=reparam,
    )
    assert (
        repr(sum_layer).splitlines()[0] == f"SymbolicSumLayer@0x{id(sum_layer):x}("
        "scope=Scope({0, 1}), arity=1, num_units=3, layer_cls=<class 'cirkit.new.layers.inner"
        ".sum.dense.DenseLayer'>, layer_kwargs={}, reparam=ExpReparam("
    )


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
    assert (
        repr(input_layer0).splitlines()[0] == f"SymbolicInputLayer@0x{id(input_layer0):x}("
        "scope=Scope({0}), num_units=3, layer_cls=<class 'cirkit.new.layers.input.exp_family"
        ".categorical.CategoricalLayer'>, layer_kwargs={'num_categories': 5}, reparam=ExpReparam("
    )
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
    assert (
        repr(prod_layer).splitlines()[0] == f"SymbolicProductLayer@0x{id(prod_layer):x}("
        "scope=Scope({0, 1}), arity=2, num_units=9, layer_cls=<class 'cirkit.new.layers.inner"
        ".sum_product.tucker.TuckerLayer'>, layer_kwargs={})"
    )

    sum_layer = SymbolicSumLayer(
        region_node,
        (prod_layer,),
        num_units=num_units,
        layer_cls=TuckerLayer,
        layer_kwargs=sum_kwargs,
        reparam=reparam,
    )
    assert (
        repr(sum_layer).splitlines()[0] == f"SymbolicSumLayer@0x{id(sum_layer):x}("
        "scope=Scope({0, 1}), arity=1, num_units=3, layer_cls=<class 'cirkit.new.layers.inner"
        ".sum_product.tucker.TuckerLayer'>, layer_kwargs={}, reparam=ExpReparam("
    )
