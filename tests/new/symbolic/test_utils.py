from typing import Dict, Literal, Type

from cirkit.new.layers import CategoricalLayer, CPLayer, InputLayer, NormalLayer
from cirkit.new.region_graph import RegionGraph, RegionNode
from cirkit.new.reparams import EFNormalReparam, ExpReparam, Reparameterization, SoftmaxReparam
from cirkit.new.symbolic import SymbolicTensorizedCircuit
from cirkit.new.utils.type_aliases import SymbLayerCfg


def get_simple_rg() -> RegionGraph:
    rg = RegionGraph()
    node1 = RegionNode({0})
    node2 = RegionNode({1})
    region = RegionNode({0, 1})
    rg.add_partitioning(region, (node1, node2))
    return rg.freeze()


def get_symbolic_circuit_on_rg(
    rg: RegionGraph, setting: Literal["cat", "norm"] = "cat"
) -> SymbolicTensorizedCircuit:
    num_units = 4
    if setting == "cat":
        input_cls: Type[InputLayer] = CategoricalLayer
        input_kwargs = {"num_categories": 256}
        input_reparam: Type[Reparameterization] = SoftmaxReparam
    elif setting == "norm":
        input_cls = NormalLayer
        input_kwargs = {}
        input_reparam = EFNormalReparam
    else:
        assert False, "This should not happen."
    inner_cls = CPLayer
    inner_kwargs: Dict[str, None] = {}  # Avoid Any.
    reparam = ExpReparam

    return SymbolicTensorizedCircuit(
        rg,
        num_input_units=num_units,
        num_sum_units=num_units,
        input_cfg=SymbLayerCfg(
            layer_cls=input_cls, layer_kwargs=input_kwargs, reparam_factory=input_reparam
        ),
        sum_cfg=SymbLayerCfg(
            layer_cls=inner_cls, layer_kwargs=inner_kwargs, reparam_factory=reparam
        ),
        prod_cfg=SymbLayerCfg(layer_cls=inner_cls, layer_kwargs=inner_kwargs),
    )
