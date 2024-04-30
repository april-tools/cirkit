from typing import Optional

from torch import Tensor

from cirkit.backend.torch.layers.input.base import TorchInputLayer
from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.semiring import SemiringCls


class TorchLogPartitionLayer(TorchInputLayer):
    """The constant input layer, with no parameters."""

    # We still accept any Reparameterization instance for reparam, but it will be ignored.
    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        num_variables: int,
        num_output_units: int,
        *,
        num_channels: int = 1,
        value: AbstractTorchParameter,
        semiring: Optional[SemiringCls] = None,
    ) -> None:
        """Init class.

        Args:
            num_variables (int): The number of variables.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels. Defaults to 1.
            value (Optional[Reparameterization], optional): Ignored. This layer has no params.
        """
        assert value.shape == (num_output_units,)
        super().__init__(
            num_variables, num_output_units, num_channels=num_channels, semiring=semiring
        )
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        return self.semiring.from_lse_sum(self.value().expand(*x.shape[1:-1], -1))
