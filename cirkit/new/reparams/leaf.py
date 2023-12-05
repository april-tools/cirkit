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

    def materialize(self, shape: Sequence[int], /, **_kwargs: Unpack[MaterializeKwargs]) -> None:
        """Materialize the internal parameter tensors with given shape.

        The initial value of the parameter after materialization is not guaranteed, and explicit \
        initialization is expected.

        Args:
            shape (Sequence[int]): The shape of the output parameter.
            **_kwargs (Unpack[MaterializeKwargs]): Unused. See Reparameterization.materialize().
        """
        super().materialize(shape, dim=())
        self.param.materialize(self.shape)

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
