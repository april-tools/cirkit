from abc import ABC, abstractmethod
from typing import Any

import einops as E
import torch
from torch import Tensor

from cirkit.backend.torch.layers.base import TorchLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import Semiring


class TorchInnerLayer(TorchLayer, ABC):
    """The abstract base class for inner layers."""

    # __init__ is overriden here to change the default value of arity, as arity=2 is the most common
    # case for all inner layers.
    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        *,
        arity: int = 2,
        num_folds: int = 1,
        semiring: Semiring | None = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
            num_folds (int): The number of channels. Defaults to 1.
        """
        super().__init__(
            num_input_units, num_output_units, arity=arity, num_folds=num_folds, semiring=semiring
        )

    @property
    def fold_settings(self) -> tuple[Any, ...]:
        return self.num_input_units, self.num_output_units, self.arity

    def sample(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        raise TypeError(f"Sampling not implemented for {type(self)}")


class TorchProductLayer(TorchInnerLayer, ABC):
    ...


class TorchSumLayer(TorchInnerLayer, ABC):
    ...


class TorchHadamardLayer(TorchProductLayer):
    """The Hadamard product layer."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        *,
        arity: int = 2,
        num_folds: int = 1,
        semiring: Semiring | None = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units, must be the same as input.
            num_folds (int): The number of channels. Defaults to 1.
            arity (int, optional): The arity of the layer. Defaults to 2.
        """
        if arity < 2:
            raise ValueError("The arity should be at least 2")
        if num_output_units != num_input_units:
            raise ValueError(
                "The number of input and output units must be the same for Hadamard product"
            )
        super().__init__(
            num_input_units, num_output_units, arity=arity, num_folds=num_folds, semiring=semiring
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, B, Ko).
        """
        return self.semiring.prod(x, dim=1, keepdim=False)  # shape (F, H, B, K) -> (F, B, K).

    def sample(self, x: Tensor) -> tuple[Tensor, None]:
        # Concatenate samples over disjoint variables through a sum
        # x: (F, H, C, K, num_samples, D)
        samples = torch.sum(x, dim=1)  # (F, C, K, num_samples, D)
        return samples, None


class TorchKroneckerLayer(TorchProductLayer):
    """The Kronecker product layer."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        *,
        arity: int = 2,
        num_folds: int = 1,
        semiring: Semiring | None = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units, must be input**arity.
            arity (int, optional): The arity of the layer, must be 2. Defaults to 2.
            num_folds (int): The number of channels. Defaults to 1.
        """
        assert num_output_units == num_input_units**arity, (
            "The number of output units must be the number of input units raised to the power of "
            "arity for Kronecker product."
        )
        if arity != 2:
            raise NotImplementedError("Kronecker only implemented for binary product units.")
        super().__init__(
            num_input_units, num_output_units, arity=arity, num_folds=num_folds, semiring=semiring
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, B, Ko).
        """
        x0 = x[:, 0].unsqueeze(dim=-1)  # shape (F, B, Ki, 1).
        x1 = x[:, 1].unsqueeze(dim=-2)  # shape (F, B, 1, Ki).
        # shape (F, B, Ki, Ki) -> (F, B, Ko=Ki**2).
        return self.semiring.mul(x0, x1).flatten(start_dim=-2)

    def sample(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        # x: (F, H, C, K, num_samples, D)
        x0 = x[:, 0].unsqueeze(dim=3)  # (F, C, Ki, 1, num_samples, D)
        x1 = x[:, 1].unsqueeze(dim=2)  # (F, C, 1, Ki, num_samples, D)
        # shape (F, C, Ki, Ki, num_samples, D) -> (F, C, Ko=Ki**2, num_samples, D)
        x = x0 + x1
        return torch.flatten(x, start_dim=2, end_dim=3), None


class TorchDenseLayer(TorchSumLayer):
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
        assert weight.num_folds == num_folds
        assert weight.shape == (num_output_units, num_input_units)
        super().__init__(
            num_input_units, num_output_units, arity=1, num_folds=num_folds, semiring=semiring
        )
        self.weight = weight

    @property
    def config(self) -> dict[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "num_folds": self.num_folds,
        }

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
        x = x.squeeze(dim=1)  # shape (F, H=1, B, Ki) -> (F, B, Ki).
        weight = self.weight()
        return self.semiring.einsum(
            "fbi,foi->fbo", inputs=(x,), operands=(weight,), dim=-1, keepdim=True
        )  # shape (F, B, Ko).

    def sample(self, x: Tensor) -> tuple[Tensor, Tensor]:
        weight = self.weight()

        negative = torch.any(weight < 0.0)
        if negative:
            raise ValueError("Sampling only works with positive weights")

        normalized = torch.allclose(torch.sum(weight, dim=-1), torch.ones(1, device=weight.device))
        if not normalized:
            raise ValueError("Sampling only works with a normalized parametrization")

        # x: (F, H, C, K, num_samples, D)
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


class TorchMixingLayer(TorchSumLayer):
    """The sum layer for mixture among layers.

    It can also be used as a sparse sum within a layer when arity=1.
    """

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        *,
        arity: int = 2,
        num_folds: int = 1,
        weight: TorchParameter,
        semiring: Semiring | None = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units, must be the same as input.
            arity (int, optional): The arity of the layer. Defaults to 2.
            num_folds (int): The number of channels. Defaults to 1.
            weight (TorchParameter): The reparameterization for layer parameters.
        """
        assert (
            num_output_units == num_input_units
        ), "The number of input and output units must be the same for MixingLayer."
        assert weight.num_folds == num_folds
        assert weight.shape == (num_output_units, arity)
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
        # shape (F, H, B, K) -> (F, B, K).
        weight = self.weight()
        return self.semiring.einsum(
            "fhbk,fkh->fbk", inputs=(x,), operands=(weight,), dim=1, keepdim=False
        )

    def sample(self, num_samples: int, x: Tensor) -> tuple[Tensor, Tensor]:
        weight = self.weight()

        negative = torch.any(weight < 0.0)
        if negative:
            raise ValueError("Sampling only works with positive weights")

        normalized = torch.allclose(torch.sum(weight, dim=-1), torch.ones(1, device=weight.device))
        if not normalized:
            raise ValueError("Sampling only works with a normalized parametrization")

        # x: (F, H, C, K, num_samples, D)
        c = x.shape[2]
        k = x.shape[-3]
        d = x.shape[-1]
        num_samples = x.shape[-2]

        # mixing_distribution: (F, O, K)
        mixing_distribution = torch.distributions.Categorical(probs=weight)

        mixing_samples = mixing_distribution.sample((num_samples,))
        mixing_samples = E.rearrange(mixing_samples, "n f k -> f k n")
        mixing_indices = E.repeat(mixing_samples, "f k n -> f 1 c k n d", c=c, k=k, d=d)

        x = torch.gather(x, 1, mixing_indices)[:, 0]
        return x, mixing_samples
