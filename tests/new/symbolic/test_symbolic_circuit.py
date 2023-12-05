# pylint: disable=missing-function-docstring
import pytest

from cirkit.layers.input.exp_family import CategoricalLayer, ExpFamilyLayer
from cirkit.layers.sum_product import BaseCPLayer, SumProductLayer
from cirkit.new.symbolic import SymbolicInputLayer, SymbolicProductLayer, SymbolicSumLayer
from cirkit.new.symbolic.symbolic_circuit import SymbolicCircuit
from cirkit.region_graph import PartitionNode, RegionGraph, RegionNode
from cirkit.region_graph.quad_tree import QuadTree
from cirkit.reparams.leaf import ReparamExp

efamily_cls = CategoricalLayer
efamily_kwargs = {"num_categories": 256}
layer_cls = BaseCPLayer
layer_kwargs = {"rank": 1}
reparam = ReparamExp

num_units = 3


def create_simple_circuit() -> SymbolicCircuit:
    circuit = SymbolicCircuit()
    input_layer_1 = SymbolicInputLayer(
        scope=frozenset({1}),
        num_units=num_units,
        efamily_cls=efamily_cls,
        efamily_kwargs=efamily_kwargs,
    )
    input_layer_2 = SymbolicInputLayer(
        scope=frozenset({2}),
        num_units=num_units,
        efamily_cls=efamily_cls,
        efamily_kwargs=efamily_kwargs,
    )
    sum_layer = SymbolicSumLayer(
        scope=frozenset({1, 2}), num_units=num_units, layer_cls=layer_cls, layer_kwargs=layer_kwargs
    )
    product_layer = SymbolicProductLayer(
        scope=frozenset({1, 2}), num_units=num_units, layer_cls=layer_cls
    )

    circuit.add_edge(input_layer_1, product_layer)
    circuit.add_edge(input_layer_2, product_layer)
    circuit.add_edge(product_layer, sum_layer)

    return circuit


def test_add_layer() -> None:
    circuit = SymbolicCircuit()
    layer = SymbolicInputLayer(
        scope=frozenset({1}),
        num_units=num_units,
        efamily_cls=efamily_cls,
        efamily_kwargs=efamily_kwargs,
    )
    circuit.add_layer(layer)
    assert layer in circuit.layers


def test_add_edge() -> None:
    circuit = create_simple_circuit()
    output_layer = list(circuit.output_layers)[0]
    input_layers = list(circuit.input_layers)
    sum_layer = list(circuit.sum_layers)[0]
    product_layer = list(circuit.product_layers)[0]
    assert all(input_layer in product_layer.inputs for input_layer in input_layers)
    assert sum_layer in product_layer.outputs


def test_smoothness() -> None:
    circuit = SymbolicCircuit()
    input_layer = SymbolicInputLayer(
        scope=frozenset({1, 2}),
        num_units=num_units,
        efamily_cls=efamily_cls,
        efamily_kwargs=efamily_kwargs,
    )
    sum_layer = SymbolicSumLayer(
        scope=frozenset({1, 2}), num_units=num_units, layer_cls=layer_cls, layer_kwargs=layer_kwargs
    )
    product_layer = SymbolicProductLayer(
        scope=frozenset({1, 2}), num_units=num_units, layer_cls=layer_cls
    )

    circuit.add_edge(input_layer, sum_layer)
    circuit.add_edge(sum_layer, product_layer)

    assert circuit.is_smooth

    circuit = create_simple_circuit()
    assert circuit.is_smooth

    # Create a non-smooth circuit
    circuit = SymbolicCircuit()
    input_layer = SymbolicInputLayer(
        scope=frozenset({1}),
        num_units=num_units,
        efamily_cls=efamily_cls,
        efamily_kwargs=efamily_kwargs,
    )
    sum_layer = SymbolicSumLayer(
        scope=frozenset({1, 2}), num_units=num_units, layer_cls=layer_cls, layer_kwargs=layer_kwargs
    )

    circuit.add_layer(input_layer)
    circuit.add_layer(sum_layer)
    circuit.add_edge(input_layer, sum_layer)

    assert not circuit.is_smooth


def test_decomposability() -> None:
    circuit = create_simple_circuit()

    assert circuit.is_decomposable

    # Create a non-decomposable circuit
    circuit = SymbolicCircuit()
    input_layer1 = SymbolicInputLayer(
        scope=frozenset({1}),
        num_units=num_units,
        efamily_cls=efamily_cls,
        efamily_kwargs=efamily_kwargs,
    )
    input_layer2 = SymbolicInputLayer(
        scope=frozenset({2}),
        num_units=num_units,
        efamily_cls=efamily_cls,
        efamily_kwargs=efamily_kwargs,
    )
    product_layer = SymbolicProductLayer(
        scope=frozenset({1, 2, 3, 4}), num_units=num_units, layer_cls=layer_cls
    )

    circuit.add_layer(input_layer1)
    circuit.add_layer(input_layer2)
    circuit.add_layer(product_layer)

    circuit.add_edge(input_layer1, product_layer)
    circuit.add_edge(input_layer2, product_layer)

    assert not circuit.is_decomposable


def test_structured_decomposability() -> None:
    circuit = create_simple_circuit()

    assert circuit.is_structured_decomposable

    # Create a non-structured-decomposable circuit
    circuit = SymbolicCircuit()
    input_layer1 = SymbolicInputLayer(
        scope=frozenset({1}),
        num_units=num_units,
        efamily_cls=efamily_cls,
        efamily_kwargs=efamily_kwargs,
    )
    input_layer2 = SymbolicInputLayer(
        scope=frozenset({2}),
        num_units=num_units,
        efamily_cls=efamily_cls,
        efamily_kwargs=efamily_kwargs,
    )
    product_layer1 = SymbolicProductLayer(
        scope=frozenset({1, 2}), num_units=num_units, layer_cls=layer_cls
    )
    product_layer2 = SymbolicProductLayer(
        scope=frozenset({2, 3}), num_units=num_units, layer_cls=layer_cls
    )

    circuit.add_layer(input_layer1)
    circuit.add_layer(input_layer2)
    circuit.add_layer(product_layer1)
    circuit.add_layer(product_layer2)

    circuit.add_edge(input_layer1, product_layer1)
    circuit.add_edge(input_layer2, product_layer1)
    circuit.add_edge(input_layer2, product_layer2)

    assert not circuit.is_structured_decomposable


def test_compatibility() -> None:
    rg = RegionGraph()
    part = PartitionNode((1, 2, 3))
    region = RegionNode((1, 2))
    part_1_2 = PartitionNode((1, 2))
    rg.add_edge(RegionNode((1,)), part_1_2)
    rg.add_edge(RegionNode((2,)), part_1_2)
    rg.add_edge(part_1_2, region)
    rg.add_edge(region, part)
    rg.add_edge(RegionNode((3,)), part)
    rg.add_edge(part, RegionNode((1, 2, 3)))

    circuit_1 = SymbolicCircuit()
    circuit_1.from_region_graph(
        rg, layer_cls, efamily_cls, layer_kwargs, efamily_kwargs, reparam, 4, 4, 1, 1
    )

    rg_2 = RegionGraph()
    region_2 = RegionNode((1, 2))
    part_2 = PartitionNode((1, 2))
    rg_2.add_edge(RegionNode((1,)), part_2)
    rg_2.add_edge(RegionNode((2,)), part_2)
    rg_2.add_edge(part_2, region_2)

    circuit_2 = SymbolicCircuit()
    circuit_2.from_region_graph(
        rg_2, layer_cls, efamily_cls, layer_kwargs, efamily_kwargs, reparam, 5, 5, 1, 1
    )

    x_scope = circuit_1.scope & circuit_2.scope
    assert circuit_1.is_compatible(circuit_2, x_scope)
    assert circuit_2.is_compatible(circuit_1, x_scope)

    # create non-compatible circuiut_3
    rg_3 = RegionGraph()
    part_3 = PartitionNode((1, 2, 3))
    region_3 = RegionNode((1, 3))
    part_3_2 = PartitionNode((1, 3))
    rg_3.add_edge(RegionNode((1,)), part_3_2)
    rg_3.add_edge(RegionNode((3,)), part_3_2)
    rg_3.add_edge(part_3_2, region_3)
    rg_3.add_edge(region_3, part_3)
    rg_3.add_edge(RegionNode((2,)), part_3)
    rg_3.add_edge(part_3, RegionNode((1, 2, 3)))

    circuit_3 = SymbolicCircuit()
    circuit_3.from_region_graph(
        rg_3, layer_cls, efamily_cls, layer_kwargs, efamily_kwargs, reparam, 5, 5, 1, 1
    )

    x_scope = circuit_1.scope & circuit_3.scope
    assert not circuit_1.is_compatible(circuit_3, x_scope)


def test_from_region_graph():
    rg = RegionGraph()
    node1 = RegionNode((1,))
    node2 = RegionNode((2,))
    partition = PartitionNode((1, 2))
    region = RegionNode((1, 2))
    rg.add_edge(node1, partition)
    rg.add_edge(node2, partition)
    rg.add_edge(partition, region)

    circuit = SymbolicCircuit()
    circuit.from_region_graph(
        rg, layer_cls, efamily_cls, layer_kwargs, efamily_kwargs, reparam, 4, 4, 1, 1
    )

    assert len(list(circuit.layers)) == 4
    assert any(isinstance(layer, SymbolicInputLayer) for layer in circuit.input_layers)
    assert any(isinstance(layer, SymbolicSumLayer) for layer in circuit.output_layers)

    rg_2 = QuadTree(4, 4, struct_decomp=True)

    circuit_2 = SymbolicCircuit()
    circuit_2.from_region_graph(
        rg_2, layer_cls, efamily_cls, layer_kwargs, efamily_kwargs, reparam, 4, 4, 1, 1
    )

    assert len(list(circuit_2.layers)) == 46
    assert len(list(circuit_2.input_layers)) == 16
    assert len(list(circuit_2.output_layers)) == 1
    assert (circuit_2.scope) == frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
