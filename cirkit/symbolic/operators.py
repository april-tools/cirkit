from typing import Iterable, Optional

from cirkit.symbolic.layers import ConstantLayer, ExpFamilyLayer
from cirkit.utils.scope import Scope


def integrate_ef_layer(sl: ExpFamilyLayer, scope: Optional[Iterable[int]] = None) -> ConstantLayer:
    scope = Scope(scope) if scope is not None else sl.scope
    # Symbolically integrate an exponential family layer, which is a constant layer
    return ConstantLayer(sl.scope, sl.num_output_units, sl.num_channels, value=1.0)
