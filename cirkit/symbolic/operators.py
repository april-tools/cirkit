from typing import Iterable, Optional

from cirkit.symbolic.sym_layers import (
    SymbConstantLayer,
    SymbExpFamilyLayer,
    SymbInputLayer,
    SymbLayerOperation,
    SymbLayerOperator,
)
from cirkit.utils import Scope


def integrate_ef_layer(
    sl: SymbExpFamilyLayer, scope: Optional[Iterable[int]] = None
) -> SymbConstantLayer:
    # Symbolically integrate an exponential family layer
    return SymbConstantLayer(sl.scope, sl.num_units, sl.num_channels, value=1.0)


def integrate_input_layer(
    sl: SymbInputLayer, scope: Optional[Iterable[int]] = None
) -> SymbConstantLayer:
    # Fallback functional as to implement symbolic integration over any other symbolic input layer
    # Note that the operator data structure will store the relevant information,
    # i.e., the layer itself as an operand and the variables to integrate as metadata
    scope = Scope(scope) if scope is not None else sl.scope
    return SymbConstantLayer(
        sl.scope,
        sl.num_units,
        num_channels=sl.num_channels,
        operation=SymbLayerOperation(
            SymbLayerOperator.INTEGRATION, operands=(sl,), metadata=dict(scope=scope)
        ),
    )
