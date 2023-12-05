# type: ignore
# pylint: skip-file

import pytest

from cirkit.layers.input.exp_family import CategoricalLayer
from cirkit.layers.sum_product import BaseCPLayer
from cirkit.new.symbolic.symbolic_circuit import SymbolicCircuit
from cirkit.new.symbolic.symbolic_layer import (
    SymbolicInputLayer,
    SymbolicProductLayer,
    SymbolicSumLayer,
)
from cirkit.region_graph import PartitionNode, RegionGraph, RegionNode
from cirkit.region_graph.quad_tree import QuadTree
from cirkit.reparams.leaf import ReparamExp

efamily_cls = CategoricalLayer
efamily_kwargs = {"num_categories": 256}
layer_cls = BaseCPLayer
layer_kwargs = {"rank": 1}
reparam = ReparamExp

num_units = 3


def test_symbolic_circuit():
    rg = RegionGraph()
    node1 = RegionNode((1,))
    node2 = RegionNode((2,))
    partition = PartitionNode((1, 2))
    region = RegionNode((1, 2))
    rg.add_edge(node1, partition)
    rg.add_edge(node2, partition)
    rg.add_edge(partition, region)

    circuit = SymbolicCircuit(
        rg, layer_cls, efamily_cls, layer_kwargs, efamily_kwargs, reparam, 4, 4, 1, 1
    )

    assert len(list(circuit.layers)) == 4
    assert any(isinstance(layer, SymbolicInputLayer) for layer in circuit.input_layers)
    assert any(isinstance(layer, SymbolicSumLayer) for layer in circuit.output_layers)

    rg_2 = QuadTree(4, 4, struct_decomp=True)

    circuit_2 = SymbolicCircuit(
        rg_2, layer_cls, efamily_cls, layer_kwargs, efamily_kwargs, reparam, 4, 4, 1, 1
    )

    assert len(list(circuit_2.layers)) == 46
    assert len(list(circuit_2.input_layers)) == 16
    assert len(list(circuit_2.output_layers)) == 1
    assert (circuit_2.scope) == frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})


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

    circuit_1 = SymbolicCircuit(
        rg, layer_cls, efamily_cls, layer_kwargs, efamily_kwargs, reparam, 4, 4, 1, 1
    )

    rg_2 = RegionGraph()
    region_2 = RegionNode((1, 2))
    part_2 = PartitionNode((1, 2))
    rg_2.add_edge(RegionNode((1,)), part_2)
    rg_2.add_edge(RegionNode((2,)), part_2)
    rg_2.add_edge(part_2, region_2)

    circuit_2 = SymbolicCircuit(
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

    circuit_3 = SymbolicCircuit(
        rg_3, layer_cls, efamily_cls, layer_kwargs, efamily_kwargs, reparam, 5, 5, 1, 1
    )

    x_scope = circuit_1.scope & circuit_3.scope
    assert not circuit_1.is_compatible(circuit_3, x_scope)
