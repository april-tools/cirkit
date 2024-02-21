from typing import Literal, cast

import torch
from torch import Tensor

from cirkit.new.layers.inner.sum_product.sum_product import SumProductLayer
from cirkit.new.reparams import Reparameterization


class TuckerLayer(SumProductLayer):
    """The Tucker (2) layer, which is a fused dense-kronecker.

    A ternary einsum is used to fuse the sum and product.
    """

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: Literal[2] = 2,
        reparam: Reparameterization,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (Literal[2], optional): The arity of the layer, must be 2. Defaults to 2.
            reparam (Reparameterization): The reparameterization for layer parameters.
        """
        if arity != 2:
            raise NotImplementedError("Tucker (2) only implemented for binary product units.")
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        self.params = reparam
        self.materialize_params((num_output_units, num_input_units, num_input_units), dim=(1, 2))

    @classmethod
    def _infer_num_prod_units(cls, num_input_units: int, arity: int = 2) -> int:
        """Infer the number of product units in the layer based on given information.

        Args:
            num_input_units (int): The number of input units.
            arity (int, optional): The arity of the layer. Defaults to 2.

        Returns:
            int: The inferred number of product units.
        """
        # CAST: int**int is not guaranteed to be int.
        return cast(int, num_input_units**arity)

    def _forward_linear(self, x0: Tensor, x1: Tensor) -> Tensor:
        # shape (*B, I), (*B, J) -> (*B, O).
        return torch.einsum("oij,...i,...j->...o", self.params(), x0, x1)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        return self.comp_space.sum(self._forward_linear, x[0], x[1], dim=-1, keepdim=True)

    # NOTE: get_product is inherited from SumLayer. The product between TuckerLayer leads to the
    #       Kronecker of the param. For the internal Kronecker, the arity is still 2, with each
    #       input mapped to the corresponding Kronecker'ed param axis. This method will also be
    #       called for SymbProdL, but what's returned is still correct with reparam unused.
