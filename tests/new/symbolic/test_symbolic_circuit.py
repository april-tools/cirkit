# pylint: disable=missing-function-docstring
from typing import Dict

from cirkit.new.layers import CategoricalLayer, CPLayer
from cirkit.new.region_graph import QuadTree, RegionGraph, RegionNode
from cirkit.new.reparams import ExpReparam
from cirkit.new.symbolic.symbolic_circuit import SymbolicCircuit
from cirkit.new.symbolic.symbolic_layer import SymbolicInputLayer, SymbolicSumLayer


def test_symbolic_circuit() -> None:
    input_cls = CategoricalLayer
    input_kwargs = {"num_categories": 256}
    layer_cls = CPLayer
    layer_kwargs: Dict[str, None] = {}
    reparam = ExpReparam()

    rg = RegionGraph()
    node1 = RegionNode((1,))
    node2 = RegionNode((2,))
    region = RegionNode((1, 2))
    rg.add_partitioning(region, [node1, node2])
    rg.freeze()

    circuit = SymbolicCircuit(
        rg,
        layer_cls,
        input_cls,
        layer_kwargs,
        input_kwargs,
        reparam=reparam,
        num_inner_units=4,
        num_input_units=4,
        num_classes=1,
    )

    assert len(list(circuit.layers)) == 4
    # Ignore: SymbolicInputLayer contains Any.
    assert all(
        isinstance(layer, SymbolicInputLayer)  # type: ignore[misc]
        for layer in circuit.input_layers
    )
    # Ignore: SymbolicSumLayer contains Any.
    assert all(
        isinstance(layer, SymbolicSumLayer) for layer in circuit.output_layers  # type: ignore[misc]
    )

    rg_2 = QuadTree((4, 4), struct_decomp=True)

    circuit_2 = SymbolicCircuit(
        rg_2,
        layer_cls,
        input_cls,
        layer_kwargs,
        input_kwargs,
        reparam=reparam,
        num_inner_units=4,
        num_input_units=4,
        num_classes=1,
    )

    assert len(list(circuit_2.layers)) == 46
    assert len(list(circuit_2.input_layers)) == 16
    assert len(list(circuit_2.output_layers)) == 1
    assert (circuit_2.scope) == frozenset(range(16))
