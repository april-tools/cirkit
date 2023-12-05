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
