from typing import Callable, Sequence, final
from typing_extensions import Unpack  # TODO: in typing from 3.12 for Unpack[dict]

import torch
from torch import Tensor, nn

from cirkit.new.reparams.reparam import Reparameterization
from cirkit.new.utils.type_aliases import MaterializeKwargs


# The LeafReparam only holds the tensor. Everything else should be a (unary) composed reparam.
# The @final is to prevent inheritance from LeafReparam.
@final
class LeafReparam(Reparameterization):
    """The leaf in reparameterizations that holds the parameter Tensor."""

    def __init__(self) -> None:
        """Init class."""
        super().__init__()
        self.param = nn.UninitializedParameter()

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""
        return self.param.dtype

    @property
    def device(self) -> torch.device:
        """The device of the output parameter."""
        return self.param.device

    def materialize(self, shape: Sequence[int], /, **_kwargs: Unpack[MaterializeKwargs]) -> bool:
        """Materialize the internal parameter tensors with given shape.

        If it is already materialized, False will be returned to indicate no materialization. \
        However, a second call to materialize must give the same config, so that the underlying \
        params can indeed be reused.

        The initial value of the parameter after materialization is not guaranteed, and explicit \
        initialization is expected.

        Args:
            shape (Sequence[int]): The shape of the output parameter.
            **_kwargs (Unpack[MaterializeKwargs]): Unused. See Reparameterization.materialize().

        Returns:
            bool: Whether the materialization is done.
        """
        if not super().materialize(shape, dim=()):
            return False
        # Not materialized before, i.e., self.param is still nn.UninitializedParameter.
        self.param.materialize(self.shape)
        return True

    def initialize(self, initializer_: Callable[[Tensor], Tensor]) -> None:
        """Initialize the internal parameter tensors with the given initializer.

        Initialization will cause error if not materialized first.

        Args:
            initializer_ (Callable[[Tensor], Tensor]): A function that can initialize a tensor \
                inplace while also returning the value.
        """
        initializer_(self.param)

    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        return self.param
