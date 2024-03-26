from typing import Optional

from cirkit.newer.symbolic.layers import SymbInputLayer
from cirkit.newer.symbolic.symb_op import SymbLayerOperation
from cirkit.utils import Scope


class SymbConstantLayer(SymbInputLayer):
    def __init__(
        self,
        scope: Scope,
        num_units: int,
        num_channels: int,
        operator: Optional[SymbLayerOperation] = None,
        value: Optional[float] = None
    ):
        assert operator is not None or value is not None, \
            "Eiether 'operator' or 'value' must be specified to construct a constant layer"
        super().__init__(scope, num_units, num_channels, operator=operator)
        self.value = value
