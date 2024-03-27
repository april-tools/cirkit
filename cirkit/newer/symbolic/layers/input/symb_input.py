from typing import List, Optional

from cirkit.newer.symbolic.layers.symb_layer import SymbLayer
from cirkit.newer.symbolic.symb_op import SymbLayerOperation
from cirkit.utils import Scope


class SymbInputLayer(SymbLayer):
    """The (abstract???) base class for symbolic input layers."""

    def __init__(
        self,
        scope: Scope,
        num_units: int,
        num_channels: int,
        operation: Optional[SymbLayerOperation] = None,
    ):
        super().__init__(scope, num_units, operation=operation, inputs=[])
        self.num_channels = num_channels
