from typing import Optional

from cirkit.newer.symbolic.layers import SymbInputLayer
from cirkit.newer.symbolic.symb_op import SymbLayerOperation
from cirkit.utils import Scope


class SymbExpFamilyLayer(SymbInputLayer):
    def __init__(
        self,
        scope: Scope,
        num_units: int,
        num_channels: int,
        operator: Optional[SymbLayerOperation] = None,
    ):
        super().__init__(scope, num_units, num_channels, operator=operator)
