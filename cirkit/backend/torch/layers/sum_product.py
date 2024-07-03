from abc import ABC
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from cirkit.backend.torch.layers.inner import TorchInnerLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import SemiringCls


class TorchSumProductLayer(TorchInnerLayer, ABC):
    @property
    def config(self) -> Dict[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
            "num_folds": self.num_folds,
        }


class TorchTuckerLayer(TorchSumProductLayer):
    """The Tucker (2) layer, which is a fused dense-kronecker.

    A ternary einsum is used to fuse the sum and product.
    """

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        *,
        num_folds: int = 1,
        weight: TorchParameter,
        semiring: Optional[SemiringCls] = None,
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
        assert weight.num_folds == num_folds
        assert weight.shape == (num_output_units, num_input_units * num_input_units)
        super().__init__(
            num_input_units, num_output_units, arity=arity, num_folds=num_folds, semiring=semiring
        )
        self.weight = weight

    @property
    def params(self) -> Dict[str, TorchParameter]:
        return dict(weight=self.weight)

    def _forward_impl(self, x1: Tensor, x2: Tensor) -> Tensor:
        # shape (*B, I), (*B, J) -> (*B, O).
        w = self.weight()
        w = w.view(-1, self.num_output_units, self.num_input_units, self.num_input_units)
        return torch.einsum("foij,f...i,f...j->f...o", w, x1, x2)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, *B, Ko).
        """
        return self.semiring.sum(self._forward_impl, x[:, 0], x[:, 1], dim=-1, keepdim=True)


class TorchCPLayer(TorchSumProductLayer):
    """The Candecomp Parafac (collapsed) layer, which is a fused dense-hadamard.

    The fusion actually does not gain anything, and is just a plain connection. We don't because \
    it cannot save computation but enforced the product into linear space, which might be worse \
    numerically.
    """

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        *,
        num_folds: int = 1,
        weight: TorchParameter,
        semiring: Optional[SemiringCls] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
            weight (TorchParameter): The reparameterization for layer parameters.
        """
        assert weight.num_folds == num_folds
        assert weight.shape == (num_output_units, num_input_units)
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            semiring=semiring,
        )
        self.weight = weight

    @property
    def params(self) -> Dict[str, TorchParameter]:
        return dict(weight=self.weight)

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.einsum("foi,f...i->f...o", self.weight(), x)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, *B, Ko).
        """
        x = self.semiring.prod(x, dim=1)  # (F, *B, Ki)
        return self.semiring.sum(self._forward_impl, x, dim=-1, keepdim=True)  # (F, *B, Ko)
