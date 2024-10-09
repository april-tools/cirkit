from abc import ABC
from typing import Any

import einops as E
import torch
from torch import Tensor

from cirkit.backend.torch.layers import TorchInnerLayer, TorchSumLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import Semiring


class TorchSumProductLayer(TorchInnerLayer, ABC):
    @property
    def config(self) -> dict[str, Any]:
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
        semiring: Semiring | None = None,
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
    def params(self) -> dict[str, TorchParameter]:
        return dict(weight=self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, B, Ko).
        """
        weight = self.weight().view(
            -1, self.num_output_units, self.num_input_units, self.num_input_units
        )
        return self.semiring.einsum(
            "fbi,fbj,foij->fbo",
            operands=(weight,),
            inputs=(x[:, 0], x[:, 1]),
            dim=-1,
            keepdim=True,
        )


class TorchCPTLayer(TorchSumProductLayer):
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
        semiring: Semiring | None = None,
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
            num_input_units,
            num_output_units,
            arity=arity,
            num_folds=num_folds,
            semiring=semiring,
        )
        self.weight = weight

    @property
    def params(self) -> dict[str, TorchParameter]:
        return dict(weight=self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, B, Ko).
        """
        x = self.semiring.prod(x, dim=1, keepdim=False)  # (F, B, Ki)
        weight = self.weight()
        return self.semiring.einsum(
            "fbi,foi->fbo", inputs=(x,), operands=(weight,), dim=-1, keepdim=True
        )

    def sample(self, x: Tensor) -> tuple[Tensor, Tensor]:
        weight = self.weight()
        negative = torch.any(weight < 0.0)
        if negative:
            raise ValueError("Sampling only works with positive weights")
        normalized = torch.allclose(torch.sum(weight, dim=-1), torch.ones(1, device=weight.device))
        if not normalized:
            raise ValueError("Sampling only works with a normalized parametrization")

        # x: (F, H=1, C, K, num_samples, D)
        x = torch.sum(x, dim=1, keepdim=True)  # (F, H=1, C, K, num_samples, D)

        c = x.shape[2]
        d = x.shape[-1]
        num_samples = x.shape[-2]

        # mixing_distribution: (F, O, K)
        mixing_distribution = torch.distributions.Categorical(probs=weight)

        mixing_samples = mixing_distribution.sample((num_samples,))
        mixing_samples = E.rearrange(mixing_samples, "n f o -> f o n")
        mixing_indices = E.repeat(mixing_samples, "f o n -> f a c o n d", a=self.arity, c=c, d=d)

        x = torch.gather(x, dim=-3, index=mixing_indices)
        x = x[:, 0]
        return x, mixing_samples


class TorchTensorDotLayer(TorchSumLayer):
    """The sum layer for dense sum within a layer."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        *,
        num_folds: int = 1,
        weight: TorchParameter,
        semiring: Semiring | None = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_outpfrom functools import cached_propertyut_units (int): The number of output units.
            num_folds (int): The number of channels. Defaults to 1.
            weight (TorchParameter): The reparameterization for layer parameters.
        """
        num_contract_units = weight.shape[1]
        num_batch_units = num_input_units // num_contract_units
        assert weight.num_folds == num_folds
        assert num_input_units % weight.shape[1] == 0
        assert num_output_units == weight.shape[0] * num_batch_units
        super().__init__(
            num_input_units, num_output_units, arity=1, num_folds=num_folds, semiring=semiring
        )
        self._num_contract_units = num_contract_units
        self._num_batch_units = num_batch_units
        self.weight = weight

    @property
    def config(self) -> dict[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "num_folds": self.num_folds,
        }

    @property
    def fold_settings(self) -> tuple[Any, ...]:
        return *super().fold_settings, self._num_batch_units

    @property
    def params(self) -> dict[str, TorchParameter]:
        return dict(weight=self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, B, Ko).
        """
        # x: (F, H=1, B, Ki) -> (F, B, Ki)
        x = x.squeeze(dim=1)
        # x: (F, B, Ki) -> (F, B, Kj, Kq) -> (F, B, Kq, Kj)
        x = x.view(x.shape[0], x.shape[1], self._num_contract_units, self._num_batch_units)
        x = x.permute(0, 1, 3, 2)
        # weight: (F, Kk, Kj)
        weight = self.weight()
        # y: (F, B, Kq, Kj)
        y = self.semiring.einsum(
            "fbqj,fkj->fbqk", inputs=(x,), operands=(weight,), dim=-1, keepdim=True
        )
        # return y: (F, B, Kq * Kj) = (F, B, Ko)
        return y.view(y.shape[0], y.shape[1], self.num_output_units)
