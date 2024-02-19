from typing import Callable, Optional, Sequence, Union, final

import torch
from torch import Tensor, nn

from cirkit.new.reparams.reparam import Reparameterization


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

    def materialize(
        self,
        shape: Sequence[int],
        /,
        *,
        dim: Union[int, Sequence[int]],
        initializer_: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> bool:
        """Materialize the internal parameter tensor with given shape and initialize if required.

        Materialization (and optionally initialization) is only executed if it's not materialized \
        yet. Otherwise this function will become a silent no-op, providing safe reuse of the same \
        reparam. However, the arguments must be the same among re-materialization attempts, to \
        make sure the reuse is consistent.

        The normalization dim is ignored for LeafReparam and normalization should be done in \
        another reparam what contains transformation.

        If an initializer_ is provided, it will be used to fill the initial value. If no \
        initializer is given, the internal storage will contain random memory.

        Args:
            shape (Sequence[int]): The shape of the output parameter.
            dim (Union[int, Sequence[int]]): Ignored. This reparam is not normalized.
            initializer_ (Optional[Callable[[Tensor], Tensor]], optional): The function that \
                initialize a Tensor inplace while also returning the value. Leave default for no \
                initialization. Defaults to None.

        Returns:
            bool: Whether the materialization is actually performed.
        """
        if not super().materialize(shape, dim=()):  # super() does not use initializer_.
            return False

        # Not materialized before, i.e., self.param is still nn.UninitializedParameter.
        self.param.materialize(self.shape)
        if initializer_ is not None:
            self.initialize(initializer_)
        return True

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
        initializer_(self.param)

    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        return self.param
