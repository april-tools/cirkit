from typing import Callable, Optional

from torch import Tensor

from cirkit.new.layers.inner.inner import InnerLayer


class ProductLayer(InnerLayer):
    """The abstract base class for product layers."""

    # NOTE: We don't change the __init__ of InnerLayer here. We still accept any Reparameterization
    #       instance in ProductLayer, but it will be ignored.

    # NOTE: We need to annotate as Optional instead of None to make SumProdL work.
    @property
    def _default_initializer_(self) -> Optional[Callable[[Tensor], Tensor]]:
        """The default inplace initializer for the parameters of this layer.

        No initialization, as ProductLayer has no parameters.
        """
        return None
