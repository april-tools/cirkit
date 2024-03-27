from typing import Optional

from cirkit.utils import Scope
from cirkit.symbolic.layers.symb_layer import SymbLayer
from cirkit.symbolic.symb_op import SymbLayerOperation


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


class SymbExpFamilyLayer(SymbInputLayer):
    def __init__(
        self,
        scope: Scope,
        num_units: int,
        num_channels: int,
        operation: Optional[SymbLayerOperation] = None,
    ):
        super().__init__(scope, num_units, num_channels, operation=operation)


class SymbCategoricalLayer(SymbExpFamilyLayer):
    ...


class SymbConstantLayer(SymbInputLayer):
    def __init__(
        self,
        scope: Scope,
        num_units: int,
        num_channels: int,
        operation: Optional[SymbLayerOperation] = None,
        value: Optional[float] = None,
    ):
        assert (
            operation is not None or value is not None
        ), "Eiether 'operation' or 'value' must be specified to construct a constant layer"
        super().__init__(scope, num_units, num_channels, operation=operation)
        self.value = value

    @property
    def kwargs(self) -> dict:
        return dict(value=self.value)
