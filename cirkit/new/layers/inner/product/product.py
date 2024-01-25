from cirkit.new.layers.inner.inner import InnerLayer


class ProductLayer(InnerLayer):
    """The abstract base class for product layers."""

    # NOTE: We don't change the __init__ of InnerLayer here. We still accept any Reparameterization
    #       instance in ProductLayer, but it will be ignored.

    def reset_parameters(self) -> None:
        """Do nothing, as product layers do not have parameters."""
