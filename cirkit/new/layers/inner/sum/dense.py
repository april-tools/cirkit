from typing import Literal

import torch
from torch import Tensor

from cirkit.new.layers.inner.sum.sum import SumLayer
from cirkit.new.reparams import Reparameterization


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
            num_output_units (int): The number of output units, must be input**arity.
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
        self.params.materialize((num_output_units, num_input_units), dim=1)

        self.reset_parameters()

    def _forward_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("oi,...i->...o", self.params(), x)  # shape (*B, Ki) -> (*B, Ko).

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, K).

        Returns:
            Tensor: The output of this layer, shape (*B, K).
        """
        x = x.squeeze(dim=0)  # shape (H=1, *B, K) -> (*B, K).
        return self.comp_space.sum(self._forward_linear, x, dim=-1, keepdim=True)  # shape (*B, K).
