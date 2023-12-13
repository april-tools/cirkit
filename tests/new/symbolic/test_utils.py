# pylint: disable=missing-function-docstring,missing-return-doc
from typing import Dict

from cirkit.new.layers import CategoricalLayer, CPLayer
from cirkit.new.region_graph import RegionGraph, RegionNode
from cirkit.new.reparams import ExpReparam
from cirkit.new.symbolic import SymbolicTensorizedCircuit


def get_simple_rg() -> RegionGraph:
    rg = RegionGraph()
    node1 = RegionNode({0})
    node2 = RegionNode({1})
    region = RegionNode({0, 1})
    rg.add_partitioning(region, (node1, node2))
    return rg.freeze()


def get_symbolic_circuit_on_rg(rg: RegionGraph) -> SymbolicTensorizedCircuit:
    num_units = 4
    input_cls = CategoricalLayer
    input_kwargs = {"num_categories": 256}
    inner_cls = CPLayer
    inner_kwargs: Dict[str, None] = {}  # Avoid Any.
    reparam = ExpReparam

    return SymbolicTensorizedCircuit(
        rg,
        num_input_units=num_units,
        num_sum_units=num_units,
        input_layer_cls=input_cls,
        input_layer_kwargs=input_kwargs,
        input_reparam=reparam,
        sum_layer_cls=inner_cls,
        sum_layer_kwargs=inner_kwargs,
        sum_reparam=reparam,
        prod_layer_cls=inner_cls,
        prod_layer_kwargs=None,
    )
