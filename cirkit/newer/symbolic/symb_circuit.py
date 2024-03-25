from typing import Iterable, Optional, Type, final

from cirkit.newer.region_graph import RegionGraph
from cirkit.newer.symbolic.layers import SymbInputLayer, SymbLayer, SymbProdLayer, SymbSumLayer


# Mark this class final so that type(SymbC) is always SymbCircuit.
@final
class SymbCircuit:
    """The symbolic representation of a (tensorized) circuit."""

    def __init__(
        self,
        region_graph: RegionGraph,
        /,
        *,
        num_channels: int = 1,
        num_input_units: Optional[int] = None,
        num_sum_units: Optional[int] = None,
        num_classes: int = 1,
        input_cls: Optional[Type[SymbInputLayer]] = None,  # TODO: how to specify?
        sum_cls: Optional[Type[SymbSumLayer]] = None,
        prod_cls: Optional[Type[SymbProdLayer]] = None,
        layers: Optional[Iterable[SymbLayer]] = None,
    ) -> None:
        pass
