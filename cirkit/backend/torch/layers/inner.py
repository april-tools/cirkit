import functools
from abc import ABC
from typing import Callable, Dict, Optional

import torch
from torch import Tensor, nn

from cirkit.backend.torch.layers.base import TorchLayer
from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.semiring import SemiringCls
from cirkit.backend.torch.utils import InitializerFunc


class TorchInnerLayer(TorchLayer, ABC):
    """The abstract base class for inner layers."""

    # __init__ is overriden here to change the default value of arity, as arity=2 is the most common
    # case for all inner layers.
    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int,
        semiring: Optional[SemiringCls] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
        """
        super().__init__(num_input_units, num_output_units, arity=arity, semiring=semiring)


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
        arity: int,
        semiring: Optional[SemiringCls] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units, must be the same as input.
            arity (int, optional): The arity of the layer. Defaults to 2.
        """
        assert (
            num_output_units == num_input_units
        ), "The number of input and output units must be the same for Hadamard product."
        super().__init__(num_input_units, num_output_units, arity=arity, semiring=semiring)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        return self.semiring.prod(x, dim=0, keepdim=False)  # shape (H, *B, K) -> (*B, K).


class TorchKroneckerLayer(TorchProductLayer):
    """The Kronecker product layer."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        semiring: Optional[SemiringCls] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units, must be input**arity.
            arity (int, optional): The arity of the layer, must be 2. Defaults to 2.
        """
        assert num_output_units == num_input_units**arity, (
            "The number of output units must be the number of input units raised to the power of "
            "arity for Kronecker product."
        )
        if arity != 2:
            raise NotImplementedError("Kronecker only implemented for binary product units.")
        super().__init__(num_input_units, num_output_units, arity=arity, semiring=semiring)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        x0 = x[0].unsqueeze(dim=-1)  # shape (*B, Ki, 1).
        x1 = x[1].unsqueeze(dim=-2)  # shape (*B, 1, Ki).
        # shape (*B, Ki, Ki) -> (*B, Ko=Ki**2).
        return self.semiring.mul(x0, x1).flatten(start_dim=-2)


class TorchDenseLayer(TorchSumLayer):
    """The sum layer for dense sum within a layer."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        weight: AbstractTorchParameter,
        semiring: Optional[SemiringCls] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            weight (AbstractTorchParameter): The reparameterization for layer parameters.
        """
        assert weight.shape == (num_output_units, num_input_units)
        super().__init__(num_input_units, num_output_units, arity=1, semiring=semiring)
        self.weight = weight

    @staticmethod
    def default_initializers() -> Dict[str, InitializerFunc]:
        return dict(weight=lambda t: nn.init.uniform_(t, 0.01, 0.99))

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.einsum("oi,...i->...o", self.weight(), x)  # shape (*B, Ki) -> (*B, Ko).

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        x = x.squeeze(dim=0)  # shape (H=1, *B, Ki) -> (*B, Ki).
        return self.semiring.sum(self._forward_impl, x, dim=-1, keepdim=True)  # shape (*B, Ko).


class TorchMixingLayer(TorchSumLayer):
    """The sum layer for mixture among layers.

    It can also be used as a sparse sum within a layer when arity=1.
    """

    # TODO: do we use another name for another purpose?

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        *,
        arity: int = 2,
        weight: AbstractTorchParameter,
        semiring: Optional[SemiringCls] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units, must be the same as input.
            arity (int, optional): The arity of the layer. Defaults to 2.
            weight (AbstractTorchParameter): The reparameterization for layer parameters.
        """
        assert (
            num_output_units == num_input_units
        ), "The number of input and output units must be the same for MixingLayer."
        assert weight.shape == (num_output_units, arity)
        super().__init__(num_input_units, num_output_units, arity=arity, semiring=semiring)
        self.weight = weight

    @staticmethod
    def default_initializers() -> Dict[str, InitializerFunc]:
        return dict(weight=lambda t: nn.init.uniform_(t, 0.01, 0.99))

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.einsum("kh,h...k->...k", self.weight(), x)  # shape (H, *B, K) -> (*B, K).

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        # shape (H, *B, K) -> (*B, K).
        return self.semiring.sum(self._forward_impl, x, dim=0, keepdim=False)
