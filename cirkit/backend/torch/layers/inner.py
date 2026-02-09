from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import einops as E
import torch
from torch import Tensor

from cirkit.backend.torch.layers.base import TorchLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import Semiring, SumProductSemiring


class TorchInnerLayer(TorchLayer, ABC):
    """The abstract base class for inner layers, i.e., either sum or product layers."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        *,
        semiring: Semiring | None = None,
        num_folds: int = 1,
    ):
        """Initialize an inner layer.

        Args:
            num_input_units: The number of input units.
            num_output_units: The number of output units.
            arity: The arity of the layer.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].
            num_folds: The number of channels.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            semiring=semiring,
            num_folds=num_folds,
        )

    @property
    def fold_settings(self) -> tuple[Any, ...]:
        pshapes = [(n, p.shape) for n, p in self.params.items()]
        return *self.config.items(), *pshapes

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Invoke the forward function.

        Args:
            x: The tensor input to this layer, having shape $(F, H, B, K_i)$, where $F$
                is the number of folds, $H$ is the arity, $B$ is the batch size, and
                $K_i$ is the number of input units.

        Returns:
            Tensor: The tensor output of this layer, having shape $(F, B, K_o)$, where $K_o$
                is the number of output units.
        """

    def sample(self, x: Tensor, ev_score: Tensor | None = None, ev_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Perform a forward sampling step.

        Args:
            x: A tensor representing the input variable assignments, having shape
                $(F, H, C, K, N, D)$, where $F$ is the number of folds, $H$ is the arity,
                $C$ is the number of channels, $K$ is the numbe rof input units, $N$ is the number
                of samples, $D$ is the number of variables.
            ev_score: A tensor of size $(F, H, C, K, N, D)$ with the evidence scores.
            ev_mask: A tensor of the same size as ev_score indicating what values are observed.

        Returns:
            Tensor: A new tensor representing the new variable assignements the layers gives
                as output.

        Raises:
            TypeError: If sampling is not supported by the layer.
        """
        raise TypeError(f"Sampling not implemented for {type(self)}")


class TorchHadamardLayer(TorchInnerLayer):
    """The Hadamard product layer, which computes an element-wise (or Hadamard) product of
    the input vectors it receives as inputs.
    See the symbolic [HadamardLayer][cirkit.symbolic.layers.HadamardLayer] for more details.
    """

    def __init__(
        self,
        num_input_units: int,
        arity: int = 2,
        *,
        semiring: Semiring | None = None,
        num_folds: int = 1,
    ):
        """Initialize a Hadamard product layer.

        Args:
            num_input_units: The number of input units, which is equal to the number of
                output units.
            arity: The arity of the layer.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].
            num_folds: The number of channels.

        Raises:
            ValueError: If the arity is not at least 2.
            ValueError: If the number of input units is not the same as the number of output units.
        """
        if arity < 2:
            raise ValueError("The arity should be at least 2")
        super().__init__(
            num_input_units, num_input_units, arity=arity, semiring=semiring, num_folds=num_folds
        )

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "arity": self.arity,
        }

    def forward(self, x: Tensor) -> Tensor:
        return self.semiring.prod(x, dim=1, keepdim=False)  # shape (F, H, B, K) -> (F, B, K).

    @torch.no_grad()
    def sample(self, x: Tensor, ev_score: Tensor | None = None, ev_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None, Tensor | None]:
        # Concatenate samples over disjoint variables through a sum
        # x: (F, H, K, num_samples, D)
        # All elements but 1 in the H axis will be "1" (in the semiring)
        x = self.semiring.prod(x, dim=1)  # (F, K, num_samples, D)
        if ev_score is None:
            return x, None, None

        ev_score = self.semiring.prod(ev_score, dim=1)
        ev_mask = ev_mask.any(dim=1)
        return x, ev_score, ev_mask


class TorchKroneckerLayer(TorchInnerLayer):
    """The Kronecker product layer, which computes the Kronecker product of the input vectors
    it receives as input.
    See the symbolic [KroneckerLayer][cirkit.symbolic.layers.KroneckerLayer] for more details.
    """

    def __init__(
        self,
        num_input_units: int,
        arity: int = 2,
        *,
        semiring: Semiring | None = None,
        num_folds: int = 1,
    ):
        """Initialize a Kronecker product layer.

        Args:
            num_input_units: The number of input units. The number of output units is the power of
                the number of input units to the arity.
            arity: The arity of the layer. Defaults to 2 (which is the only supported arity).
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].
            num_folds: The number of channels.

        Raises:
            ValueError: If the number of input units is not the same as the number of output units.
        """
        super().__init__(
            num_input_units,
            num_input_units**arity,
            arity=arity,
            semiring=semiring,
            num_folds=num_folds,
        )

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "arity": self.arity,
        }

    def forward(self, x: Tensor) -> Tensor:
        # x: (F, H, B, Ki)
        y0 = x[:, 0]
        for i in range(1, x.shape[1]):
            y0 = y0.unsqueeze(dim=-1)  # (F, B, K, 1).
            y1 = x[:, i].unsqueeze(dim=-2)  # (F, B, 1, Ki).
            # y0: (F, B, K=K * Ki).
            y0 = torch.flatten(self.semiring.mul(y0, y1), start_dim=-2)
        # y0: (F, B, Ko=Ki ** arity)
        return y0

    @torch.no_grad()
    def sample(self, x: Tensor, ev_score: Tensor | None = None, ev_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None, Tensor | None]:
        # x:        (F, H, K, num_samples, E?, D)
        # evidence: (F, H, K, num_samples, E, D)
        y0 = x[:, 0]
        ev_score = ev_score[:, :, :1, :1] # ev_score should be exactly the same for K and num_samples
        ev_mask = ev_mask[:, :, :1, :1] # ev_mask should be exactly the same for K and num_samples
        for i in range(1, x.shape[1]):
            y0 = y0.unsqueeze(dim=3)  # (F, K, 1, num_samples, D)
            y1 = x[:, i].unsqueeze(dim=2)  # (F, 1, Ki, num_samples, D)
            y0 = torch.flatten(self.semiring.mul(y0, y1), start_dim=2, end_dim=3)
        # y0: (F, Ko=Ki ** arity, num_samples, D)
        if ev_score is None:
            return y0, None, None

        raise NotImplementedError("This needs to be properly implemented :)")
        return y0, ev_score, ev_mask


class TorchSumLayer(TorchInnerLayer):
    """The sum layer torch implementation.
    See the symbolic [SumLayer][cirkit.symbolic.layers.SumLayer] for more details."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        *,
        weight: TorchParameter,
        semiring: Semiring | None = None,
        num_folds: int = 1,
    ):
        r"""Initialize a sum layer.

        Args:
            num_input_units: The number of input units.
            num_output_units: The number of output units.
            arity: The arity of the layer.
            weight: The weight parameter, which must have shape $(F, K_o, K_i\cdot H)$,
                where $F$ is the number of folds, $K_o$ is the number of output units,
                   $K_i$ is the number of input units, and $H$ is the arity.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].
            num_folds: The number of channels.

        Raises:
            ValueError: If the arity is not a positive integer.
            ValueError: If the arity, the number of input and output units are incompatible with the
                shape of the weight parameter.
        """
        if arity < 1:
            raise ValueError("The arity must be a positive integer")
        super().__init__(
            num_input_units, num_output_units, arity=arity, semiring=semiring, num_folds=num_folds
        )
        if not self._valid_weight_shape(weight):
            raise ValueError(
                f"Expected number of folds {self.num_folds} "
                f"and shape {self._weight_shape} for 'weight', found"
                f"{weight.num_folds} and {weight.shape}, respectively"
            )
        self.weight = weight

    def _valid_weight_shape(self, w: TorchParameter) -> bool:
        if w.num_folds != self.num_folds:
            return False
        return w.shape == self._weight_shape

    @property
    def _weight_shape(self) -> tuple[int, ...]:
        return self.num_output_units, self.num_input_units * self.arity

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
        # x: (F, H, B, Ki) -> (F, B, H * Ki)
        # weight: (F, Ko, H * Ki)
        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        weight = self.weight()
        return self.semiring.einsum(
            "fbi,foi->fbo", inputs=(x,), operands=(weight,), dim=-1, keepdim=True
        )  # shape (F, B, K_o).

    @torch.no_grad()
    def sample(self, x: Tensor, ev_score: Tensor | None = None, ev_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None, Tensor | None]:
        # F = Fold, H = Arity, Ki = Input Dim, Ko = Output Dim, B = Batch size, E = Evidence size, D = Data dim
        # x:        (F, H, Ki, B, E?, D)
        # ev_score: (F, H, Ki, B, E, D)
        # weight:   (F, Ko, H * Ki)
        weight = self.weight()
        if ev_score is not None:
            # The ev_score is repeated for all B, so just pick the first
            # Replace observations (samples) with 1s (in the semiring space)
            score = ev_score.select(3, 0).masked_fill(~ev_mask.select(3, 0), self.semiring.multiplicative_identity)
            score = self.semiring.prod(score, dim=-1).movedim(-1, 1).flatten(start_dim=2)
            weight = self.semiring.map_from(weight, SumProductSemiring)
            weight = self.semiring.mul(weight.unsqueeze(2), score.unsqueeze(1))
            weight = self.semiring.div(weight, self.semiring.sum(weight, dim=-1, keepdim=True))
            weight = SumProductSemiring.map_from(weight, self.semiring)
            
        negative = torch.any(weight < 0.0)
        normalized = torch.allclose(torch.sum(weight, dim=-1), torch.ones(1, device=weight.device))
        if negative or not normalized:
            raise TypeError("Sampling in sum layers only works with positive weights summing to 1")

        # x: (F, H, Ki, B, E?, D) -> (F, H * Ki, B, E?, D)
        x = x.flatten(1, 2)

        num_samples = x.shape[2]
        d = x.shape[-1]

        # mixing_distribution: (F, Ko, E?, H * Ki)
        mixing_distribution = torch.distributions.Categorical(probs=weight)

        # mixing_samples: (B, F, Ko, E?) -> (F, Ko, E?, B)
        mixing_samples = mixing_distribution.sample((num_samples,))
        mixing_samples = mixing_samples.movedim(0, 2)

        # mixing_indices: (F, Ko, E?, B, D)
        mixing_indices = mixing_samples.unsqueeze(-1).repeat([1] * len(mixing_samples.shape) + [d])

        # x: (F, E?, Ko, B, D)
        # x: (F, Ko, B, E?, D) 
        x = torch.gather(x, dim=1, index=mixing_indices)

        # (F, Ko, E, D)
        if ev_score is not None:
            ev_score = ev_score.flatten(1, 2)
            ev_score = ev_score[:, :1].expand_as(x)

            ev_mask = ev_mask.flatten(1, 2)
            ev_mask = ev_mask[:, :1].expand_as(x)

        return x, ev_score, ev_mask, mixing_samples
        