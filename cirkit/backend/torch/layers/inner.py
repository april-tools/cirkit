from abc import ABC
from collections.abc import Mapping
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
        arity: int = 2,
        *,
        semiring: Semiring | None = None,
        num_folds: int = 1,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
            num_folds (int): The number of channels. Defaults to 1.
        """
        super().__init__(
            num_input_units, num_output_units, arity=arity, semiring=semiring, num_folds=num_folds
        )

    @property
    def fold_settings(self) -> tuple[Any, ...]:
        pshapes = [(n, p.shape) for n, p in self.params.items()]
        return *self.config.items(), *pshapes

    def sample(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        raise TypeError(f"Sampling not implemented for {type(self)}")


class TorchHadamardLayer(TorchInnerLayer):
    """The Hadamard product layer."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        *,
        semiring: Semiring | None = None,
        num_folds: int = 1,
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
            num_input_units, num_output_units, arity=arity, semiring=semiring, num_folds=num_folds
        )

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
        }

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
        x = torch.sum(x, dim=1)  # (F, C, K, num_samples, D)
        return x, None


class TorchKroneckerLayer(TorchInnerLayer):
    """The Kronecker product layer."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        *,
        semiring: Semiring | None = None,
        num_folds: int = 1,
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
            num_input_units, num_output_units, arity=arity, semiring=semiring, num_folds=num_folds
        )

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
        }

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


class TorchSumLayer(TorchInnerLayer):
    """The sum layer."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        *,
        weight: TorchParameter,
        semiring: Semiring | None = None,
        num_folds: int = 1,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            weight (TorchParameter): The reparameterization for layer parameters.
            num_folds (int): The number of channels. Defaults to 1.
        """
        assert weight.num_folds == num_folds
        assert weight.shape == (num_output_units, arity * num_input_units)
        super().__init__(
            num_input_units, num_output_units, arity=arity, semiring=semiring, num_folds=num_folds
        )
        self.weight = weight

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
        }

    @property
    def params(self) -> Mapping[str, TorchParameter]:
        return {"weight": self.weight}

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, B, Ko).
        """
        # x: (F, H, B, Ki) -> (F, B, H * Ki)
        # weight: (F, Ko, H * Ki)
        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
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

        # x: (F, H, C, Ki, num_samples, D) -> (F, C, H * Ki, num_samples, D)
        x = x.permute(0, 2, 1, 3, 4, 5).flatten(2, 3)
        c = x.shape[1]
        num_samples = x.shape[3]
        d = x.shape[4]

        # mixing_distribution: (F, Ko, H * Ki)
        mixing_distribution = torch.distributions.Categorical(probs=weight)

        # mixing_samples: (num_samples, F, Ko) -> (F, Ko, num_samples)
        mixing_samples = mixing_distribution.sample((num_samples,))
        mixing_samples = E.rearrange(mixing_samples, "n f k -> f k n")

        # mixing_indices: (F, C, Ko, num_samples, D)
        mixing_indices = E.repeat(mixing_samples, "f k n -> f c k n d", c=c, d=d)

        # x: (F, C, Ko, num_samples, D)
        x = torch.gather(x, dim=2, index=mixing_indices)
        return x, mixing_samples
