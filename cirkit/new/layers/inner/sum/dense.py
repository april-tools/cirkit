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

    @classmethod
    def _infer_num_prod_units(cls, num_input_units: int, arity: int = 2) -> Literal[0]:
        """Infer the number of product units in the layer based on given information.

        This layer has no product units. This method is only for interface compatibility.

        Args:
            num_input_units (int): The number of input units.
            arity (int, optional): The arity of the layer. Defaults to 2.

        Returns:
            Literal[0]: Sum layers have 0 product units.
        """
        return 0

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

    # NOTE: get_product is inherited from SumLayer. The product between DesnLayer leads to the
    #       Kronecker of the param.
