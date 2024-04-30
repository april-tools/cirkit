from typing import Callable, Tuple, final

import torch
from torch import Tensor, nn

from cirkit.backend.torch.params.base import AbstractTorchParameter


# The LeafReparam only holds the tensor. Everything else should be a (unary) composed reparam.
# The @final is to prevent inheritance from LeafReparam.
@final
class TorchParameter(AbstractTorchParameter):
    """The leaf in reparameterizations that holds the parameter Tensor."""

    def __init__(self, *shape: int) -> None:
        """Init class."""
        super().__init__()
        self.ptensor = nn.Parameter(torch.empty(*shape), requires_grad=True)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.ptensor.shape

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""
        return self.ptensor.dtype

    @property
    def device(self) -> torch.device:
        """The device of the output parameter."""
        return self.ptensor.device

    @torch.no_grad()
    def initialize(self, initializer_: Callable[[Tensor], Tensor]) -> None:
        """Initialize the internal parameter tensor with the given initializer.

        This can only be called after materialization and will always overwrite whatever is \
        already in the internal param. To safely provide an initial value to a possibly reused \
        reparam, initialize through materialize() instead.

        The provided initializer_ is expected to provide an initial value which will be filled \
        into the underlying tensor.

        Args:
            initializer_ (Callable[[Tensor], Tensor]): The function that initialize a Tensor \
                inplace while also returning the value.
        """
        initializer_(self.ptensor)

    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        return self.ptensor
