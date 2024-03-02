from typing import Literal

import torch
from torch import Tensor

from cirkit.new.layers.inner.sum.sum import SumLayer
from cirkit.new.reparams import Reparameterization


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

    # NOTE: get_product is inherited from SumLayer. The product between MixingLayer leads to the
    #       Kronecker of the param, with the arity expanded to the product of arities.
