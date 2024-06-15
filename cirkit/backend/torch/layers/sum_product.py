from typing import Literal, cast

import torch
from torch import Tensor

from cirkit.backend.torch.layers.inner import TorchDenseLayer, TorchHadamardLayer, TorchInnerLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter


class TorchCPLayer(TorchInnerLayer):
    """The Candecomp Parafac (collapsed) layer, which is a fused dense-hadamard.

    The fusion actually does not gain anything, and is just a plain connection. We don't because \
    it cannot save computation but enforced the product into linear space, which might be worse \
    numerically.
    """

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        weight: TorchParameter,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
            weight (TorchParameter): The reparameterization for layer parameters.
        """
        super().__init__(
            num_input_units=num_input_units, num_output_units=num_output_units, arity=arity
        )

        self.prod_layer = TorchHadamardLayer(
            num_input_units=num_input_units, num_output_units=num_input_units, arity=arity
        )
        self.sum_layer = TorchDenseLayer(
            num_input_units=num_input_units, num_output_units=num_output_units, weight=weight
        )
        # The parameters belong to DenseLayer so we don't handle it here.

    @classmethod
    def _infer_num_prod_units(cls, num_input_units: int, arity: int = 2) -> int:
        """Infer the number of product units in the layer based on given information.

        Args:
            num_input_units (int): The number of input units.
            arity (int, optional): The arity of the layer. Defaults to 2.

        Returns:
            int: The inferred number of product units.
        """
        return num_input_units

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        # shape (H, *B, Ki) -> (*B, Ki) -> (H, *B, Ki) -> (*B, Ko).
        return self.sum_layer(self.prod_layer(x).unsqueeze(dim=0))


class TorchTuckerLayer(TorchInnerLayer):
    """The Tucker (2) layer, which is a fused dense-kronecker.

    A ternary einsum is used to fuse the sum and product.
    """

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: Literal[2] = 2,
        weight: TorchParameter,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (Literal[2], optional): The arity of the layer, must be 2. Defaults to 2.
            weight (TorchParameter): The reparameterization for layer parameters.
        """
        if arity != 2:
            raise NotImplementedError("Tucker (2) only implemented for binary product units.")
        super().__init__(
            num_input_units=num_input_units, num_output_units=num_output_units, arity=arity
        )
        self.weight = weight

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
        return torch.einsum("oij,...i,...j->...o", self.weight(), x0, x1)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        return self.semiring.sum(self._forward_linear, x[0], x[1], dim=-1, keepdim=True)
