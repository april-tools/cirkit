import functools
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from cirkit.backend.torch.layers import TorchLayer
from cirkit.backend.torch.params.base import AbstractTorchParameter


class TorchInnerLayer(TorchLayer):
    """The abstract base class for inner layers."""

    # __init__ is overriden here to change the default value of arity, as arity=2 is the most common
    # case for all inner layers.
    def __init__(self, *, num_input_units: int, num_output_units: int, arity: int = 2) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
        """
        super().__init__(
            num_input_units=num_input_units, num_output_units=num_output_units, arity=arity
        )


class TorchProductLayer(TorchInnerLayer):
    """The abstract base class for product layers."""

    # NOTE: We don't change the __init__ of InnerLayer here. We still accept any Reparameterization
    #       instance in ProductLayer, but it will be ignored.

    # NOTE: We need to annotate as Optional instead of None to make SumProdL work.
    @property
    def _default_initializer_(self) -> Optional[Callable[[Tensor], Tensor]]:
        """The default inplace initializer for the parameters of this layer.

        No initialization, as ProductLayer has no parameters.
        """
        return None


class TorchHadamardLayer(TorchProductLayer):
    """The Hadamard product layer."""

    def __init__(self, *, num_input_units: int, num_output_units: int, arity: int = 2) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units, must be the same as input.
            arity (int, optional): The arity of the layer. Defaults to 2.
        """
        assert (
            num_output_units == num_input_units
        ), "The number of input and output units must be the same for Hadamard product."
        super().__init__(
            num_input_units=num_input_units, num_output_units=num_output_units, arity=arity
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        return self.comp_space.prod(x, dim=0, keepdim=False)  # shape (H, *B, K) -> (*B, K).


class TorchKroneckerLayer(TorchProductLayer):
    """The Kronecker product layer."""

    def __init__(self, *, num_input_units: int, num_output_units: int, arity: int = 2) -> None:
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
        super().__init__(
            num_input_units=num_input_units, num_output_units=num_output_units, arity=arity
        )

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
        return self.comp_space.mul(x0, x1).flatten(start_dim=-2)


class TorchSumLayer(TorchInnerLayer):
    """The abstract base class for sum layers."""

    # NOTE: We don't change the __init__ of InnerLayer here. Although sum layers typically have
    #       parameters, we still allow it to be optional for flexibility.

    @property
    def _default_initializer_(self) -> Callable[[Tensor], Tensor]:
        """The default inplace initializer for the parameters of this layer.

        The sum weights are initialized to U(0.01, 0.99).
        """
        return functools.partial(nn.init.uniform_, a=0.01, b=0.99)


class TorchDenseLayer(TorchSumLayer):
    """The sum layer for dense sum within a layer."""

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        param: AbstractTorchParameter,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            param (AbstractTorchParameter): The reparameterization for layer parameters.
        """
        super().__init__(
            num_input_units=num_input_units, num_output_units=num_output_units, arity=1
        )

        self.param = param

    def _forward_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("oi,...i->...o", self.param(), x)  # shape (*B, Ki) -> (*B, Ko).

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        x = x.squeeze(dim=0)  # shape (H=1, *B, Ki) -> (*B, Ki).
        return self.comp_space.sum(self._forward_linear, x, dim=-1, keepdim=True)  # shape (*B, Ko).


class TorchMixingLayer(TorchSumLayer):
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
        param: AbstractTorchParameter,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units, must be the same as input.
            arity (int, optional): The arity of the layer. Defaults to 2.
            param (AbstractTorchParameter): The reparameterization for layer parameters.
        """
        assert (
            num_output_units == num_input_units
        ), "The number of input and output units must be the same for MixingLayer."
        super().__init__(
            num_input_units=num_input_units, num_output_units=num_output_units, arity=arity
        )
        self.param = param

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
