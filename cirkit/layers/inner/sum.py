import functools
from typing import Callable, Literal

import torch
from torch import Tensor, nn

from cirkit.layers import InnerLayer
from cirkit.tensorized.reparams import Reparameterization


class SumLayer(InnerLayer):
    """The abstract base class for sum layers."""

    # NOTE: We don't change the __init__ of InnerLayer here. Although sum layers typically have
    #       parameters, we still allow it to be optional for flexibility.

    @property
    def _default_initializer_(self) -> Callable[[Tensor], Tensor]:
        """The default inplace initializer for the parameters of this layer.

        The sum weights are initialized to U(0.01, 0.99).
        """
        return functools.partial(nn.init.uniform_, a=0.01, b=0.99)


class DenseLayer(SumLayer):
    """The sum layer for dense sum within a layer."""

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: Literal[1] = 1,
        reparam: Reparameterization,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (Literal[1], optional): The arity of the layer, must be 1. Defaults to 1.
            reparam (Reparameterization): The reparameterization for layer parameters.
        """
        assert arity == 1, "DenseLayer must have arity=1. For arity>1, use MixingLayer."
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        self.params = reparam
        self.materialize_params((num_output_units, num_input_units), dim=1)

    def _forward_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("oi,...i->...o", self.params(), x)  # shape (*B, Ki) -> (*B, Ko).

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        x = x.squeeze(dim=0)  # shape (H=1, *B, Ki) -> (*B, Ki).
        return self.comp_space.sum(self._forward_linear, x, dim=-1, keepdim=True)  # shape (*B, Ko).


class MixingLayer(SumLayer):
    """The sum layer for mixture among layers.

    It can also be used as a sparse sum within a layer when arity=1.
    """

    # TODO: do we use another name for another purpose?

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        reparam: Reparameterization,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units, must be the same as input.
            arity (int, optional): The arity of the layer. Defaults to 2.
            reparam (Reparameterization): The reparameterization for layer parameters.
        """
        assert (
            num_output_units == num_input_units
        ), "The number of input and output units must be the same for MixingLayer."
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        self.params = reparam
        self.materialize_params((num_output_units, arity), dim=1)

    def _forward_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("kh,h...k->...k", self.params(), x)  # shape (H, *B, K) -> (*B, K).

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        # shape (H, *B, K) -> (*B, K).
        return self.comp_space.sum(self._forward_linear, x, dim=0, keepdim=False)
