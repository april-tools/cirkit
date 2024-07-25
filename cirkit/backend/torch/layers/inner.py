import functools
from abc import ABC
from typing import Any, Callable, Dict, Optional, Tuple, List

import torch
from torch import Tensor, nn

from cirkit.backend.torch.layers.base import TorchLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
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
        *,
        arity: int = 2,
        num_folds: int = 1,
        semiring: Optional[SemiringCls] = None,
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
        semiring: Optional[SemiringCls] = None,
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

    def extended_forward(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def sample_forward(self, num_samples: int, x: Tensor) -> Tensor:
        return self.semiring.prod(x, dim=1, keepdim=False)


class TorchKroneckerLayer(TorchProductLayer):
    """The Kronecker product layer."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        *,
        arity: int = 2,
        num_folds: int = 1,
        semiring: Optional[SemiringCls] = None,
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
        semiring: Optional[SemiringCls] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
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
        params = super().params
        params.update(weight=self.weight)
        return params

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.einsum("foi,f...i->f...o", self.weight(), x)  # shape (*B, Ki) -> (*B, Ko).

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, *B, Ko).
        """
        x = x.squeeze(dim=1)  # shape (F, H=1, *B, Ki) -> (F, *B, Ki).8
        return self.semiring.sum(self._forward_impl, x, dim=-1, keepdim=True)  # shape (F, *B, Ko).


    def extended_forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Extended forward pass is not implemented for DenseLayer.")

    def sample_forward(self, num_samples: int, x: Tensor) -> Tensor:
        x = x.squeeze(dim=1)  # shape (F, H=1, *B, Ki) -> (F, *B, Ki).8
        return self.semiring.sum(self._forward_impl, x, dim=-1, keepdim=True)  # shape (F, *B, Ko).


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
        semiring: Optional[SemiringCls] = None,
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
        params = super().params
        params.update(weight=self.weight)
        return params

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.einsum(
            "fkh,fh...k->f...k", self.weight(), x
        )  # shape (F, H, *B, K) -> (F, *B, K).

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, *B, Ko).
        """
        # shape (F, H, *B, K) -> (F, *B, K).
        return self.semiring.sum(self._forward_impl, x, dim=1, keepdim=False)

    def extended_forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Extended forward pass is not implemented for MixingLayer.")


    def sample_forward(self, num_samples: int, x: Tensor) -> Tensor:
        mixing_distribution = torch.distributions.Categorical(
            logits=self.weight()
        )  # shape (F, D, K)
        mixing_samples = mixing_distribution.sample((num_samples,))
        mixing_samples = mixing_samples.permute(1, 2, 0)
        mixing_samples = mixing_samples.unsqueeze(2).unsqueeze(-1)
        mixing_samples = mixing_samples * torch.ones_like(x[:, :1, ...])

        x = torch.gather(x, 1, mixing_samples)[:, 0]
        return x, mixing_samples

    def sample_backward(
        self,
        sample_dict: Dict[TorchLayer, List[Tensor]],
        unit_dict: Dict[TorchLayer, List[Tensor]],
        num_samples: int,
    ) -> Tensor:
        mixing_distribution = torch.distributions.Categorical(logits=self.weight())
        mixing_samples = mixing_distribution.sample((num_samples,))
        mixing_sample_weights = torch.gather(self.weight(), 1, mixing_samples.unsqueeze(1))

        idx1, idx2 = torch.unravel_index(mixing_samples, self.weight().shape)

        for i, layer in enumerate(sample_dict.keys()):
            sample_idx = sample_dict[self][idx1[i]]
            sample_dict[layer].extend(sample_idx)

            unit_idx = unit_dict[self][idx2[i]]
            unit_dict[layer].extend(unit_idx)

        return mixing_sample_weights
