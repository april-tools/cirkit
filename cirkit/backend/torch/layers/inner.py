from abc import ABC
from typing import Any, Dict, Optional, Tuple

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
        semiring: Optional[Semiring] = None,
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
    def fold_settings(self) -> Tuple[Any, ...]:
        return self.num_input_units, self.num_output_units, self.arity


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
        semiring: Optional[Semiring] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units, must be the same as input.
            num_folds (int): The number of channels. Defaults to 1.
            arity (int, optional): The arity of the layer. Defaults to 2.
        """
        assert (
            num_output_units == num_input_units
        ), "The number of input and output units must be the same for Hadamard product."
        super().__init__(
            num_input_units, num_output_units, arity=arity, num_folds=num_folds, semiring=semiring
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, *B, Ko).
        """
        return self.semiring.prod(x, dim=1, keepdim=False)  # shape (F, H, *B, K) -> (F, *B, K).


class TorchKroneckerLayer(TorchProductLayer):
    """The Kronecker product layer."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        *,
        arity: int = 2,
        num_folds: int = 1,
        semiring: Optional[Semiring] = None,
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
        *,
        num_folds: int = 1,
        weight: TorchParameter,
        semiring: Optional[Semiring] = None,
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
    def config(self) -> Dict[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "num_folds": self.num_folds,
        }

    @property
    def params(self) -> Dict[str, TorchParameter]:
        return dict(weight=self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, *B, Ko).
        """
        x = x.squeeze(dim=1)  # shape (F, H=1, *B, Ki) -> (F, *B, Ki).
        weight = self.weight()
        return self.semiring.einsum(
            "foi,f...i->f...o", operands=(weight,), inputs=(x,), dim=-1, keepdim=True
        )  # shape (F, *B, Ko).


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
        semiring: Optional[Semiring] = None,
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
    def params(self) -> Dict[str, TorchParameter]:
        return dict(weight=self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, *B, Ko).
        """
        # shape (F, H, *B, K) -> (F, *B, K).
        weight = self.weight()
        return self.semiring.einsum(
            "fkh,fh...k->f...k", operands=(weight,), inputs=(x,), dim=1, keepdim=False
        )
