from typing import Iterable, Optional

from cirkit.symbolic.sym_layers import (
    SymConstantLayer,
    SymExpFamilyLayer
)
from cirkit.utils.scope import Scope


def integrate_ef_layer(
    sl: SymExpFamilyLayer, scope: Optional[Iterable[int]] = None
) -> SymConstantLayer:
    scope = Scope(scope) if scope is not None else sl.scope
    # Symbolically integrate an exponential family layer, which is a constant layer
    return SymConstantLayer(sl.scope, sl.num_output_units, sl.num_channels, value=1.0)
