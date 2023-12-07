from typing import Optional

from cirkit.new.layers.inner.inner import InnerLayer
from cirkit.new.reparams import Reparameterization


class ProductLayer(InnerLayer):
    """The abstract base class for product layers."""

    # We still accept any Reparameterization instance for reparam, but it will be ignored.
    # TODO: this disable should be a pylint bug
    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        reparam: Optional[Reparameterization] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
            reparam (Optional[Reparameterization], optional): Ignored. This layer has no params. \
                Defaults to None.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=None,
        )

    def reset_parameters(self) -> None:
        """Do nothing as the product layers do not have parameters."""
