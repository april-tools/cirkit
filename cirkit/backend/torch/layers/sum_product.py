import torch
import einops as E

from abc import ABC
from typing import Any, Dict, Optional, Tuple

from torch import Tensor

from cirkit.backend.torch.layers.inner import TorchInnerLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import Semiring


class TorchSumProductLayer(TorchInnerLayer, ABC):
    @property
    def config(self) -> Dict[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
            "num_folds": self.num_folds,
        }

    def sample_forward(self, num_samples: int, x: Tensor) -> Tuple[Tensor, Tensor]:
        product_samples = self.prod_layer.sample_forward(num_samples, x)
        samples, mixing_samples = self.sum_layer.sample_forward(num_samples, product_samples)
        return samples, mixing_samples


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
        semiring: Optional[Semiring] = None,
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
        semiring: Optional[Semiring] = None,
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
    def params(self) -> Dict[str, TorchParameter]:
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

    def sample_forward(self, num_samples: int, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.semiring.prod(x, dim=1, keepdim=False)

        normalisation = self.weight().sum(-1).abs().mean()
        if normalisation < 1 - 1e-6 or normalisation > 1 + 1e-6:
            raise ValueError("Sampling only works with a normalised parametrisation!")

        c = x.shape[2]
        d = x.shape[-1]

        mixing_distribution = torch.distributions.Categorical(
            probs=self.weight()
        )  # shape (F, O, K)

        mixing_samples = mixing_distribution.sample((num_samples,))
        mixing_samples = E.rearrange(mixing_samples, "n f o -> f o n")
        mixing_indices = E.repeat(mixing_samples, "f o n -> f a c o n d", a=self.arity, c=c, d=d)

        x = torch.gather(x, -3, mixing_indices)
        x = x[:, 0]
        return x, mixing_samples
