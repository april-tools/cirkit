from collections.abc import Mapping
from typing import Any

import einops as E
import torch
from torch import Tensor

from cirkit.backend.torch.layers import TorchInnerLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import Semiring


class TorchTuckerLayer(TorchInnerLayer):
    """The Tucker layer optimized implementation, leveraging a ```torch.einsum``` operation."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        *,
        weight: TorchParameter,
        semiring: Semiring | None = None,
        num_folds: int = 1,
    ):
        """Initialize a Tucker layer.

        Args:
            num_input_units: The number of input units.
            num_output_units: The number of output units.
            arity: The arity of the layer, must be 2. Defaults to 2.
            weight: The weight parameter, which must have shape $(F, K_o, K_i^2)$,
                where $F$ is the number of folds, $K_o$ is the number output units,
                and $K_i$ is the number of input units.

        Raises:
            NotImplementedError: If the arity is not equal to 2. Future versions of cirkit
                will support Tucker layers having arity greter than 2.
            ValueError: If the number of input and output units are incompatible with the
                shape of the weight parameter.
        """
        # TODO: Generalize Tucker layer to have any arity greater or equal 2
        if arity != 2:
            raise NotImplementedError("The Tucker layer is only implemented with arity=2")
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
        return self.num_output_units, self.num_input_units * self.num_input_units

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
        # weight: (F, Ko, Ki * Ki) -> (F, Ko, Ki, Ki)
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


class TorchCPTLayer(TorchInnerLayer):
    """The Candecomp transposed (CP-T) layer, which is the fusion of a sum layer and a Hadamard
    layer.
    """

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        *,
        weight: TorchParameter,
        semiring: Semiring | None = None,
        num_folds: int = 1,
    ):
        """Initialize a CP-T layer.

        Args:
            num_input_units: The number of input units.
            num_output_units: The number of output units.
            arity: The arity of the layer, must be 2. Defaults to 2.
            weight: The weight parameter, which must have shape $(F, K_o, K_i)$,
                where $F$ is the number of folds, $K_o$ is the number output units,
                and $K_i$ is the number of input units.

        Raises:
            ValueError: If the number of input and output units are incompatible with the
                shape of the weight parameter.
        """
        super().__init__(
            num_input_units,
            num_output_units,
            arity=arity,
            semiring=semiring,
            num_folds=num_folds,
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
        return self.num_output_units, self.num_input_units

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
        # x: (F, B, Ki)
        x = self.semiring.prod(x, dim=1, keepdim=False)
        # weight: (F, Ko, Ki)
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

        # x: (F, H, C, K, num_samples, D)
        x = torch.sum(x, dim=1, keepdim=True)  # (F, H=1, C, K, num_samples, D)

        c = x.shape[2]
        d = x.shape[-1]
        num_samples = x.shape[-2]

        # mixing_distribution: (F, O, K)
        mixing_distribution = torch.distributions.Categorical(probs=weight)

        mixing_samples = mixing_distribution.sample((num_samples,))
        mixing_samples = E.rearrange(mixing_samples, "n f o -> f o n")
        mixing_indices = E.repeat(mixing_samples, "f o n -> f a c o n d", a=1, c=c, d=d)

        x = torch.gather(x, dim=-3, index=mixing_indices)
        x = x[:, 0]
        return x, mixing_samples


class TorchTensorDotLayer(TorchInnerLayer):
    r"""The tensor dot layer performs the following operations.
    Let $\mathbf{x}$ be an input tensor of shape $(B, K_i)$, where $B$ is the batch size,
    and $K_i$ is the number of input uits. The tensor dot layer firstly reshapes as the tensor
    $\mathcal{Z}$ having shape $(B, K_j, K_q)$, where $K_i = K_jK_q$. Then, it computes the
    tensor $\mathcal{S}$ of shape $(B, K_q, K_k)$ as follows:

    $$
    \mathcal{S}_{bqk} = \sum_{j=1}^{K_j} w_{kj} z_{bjq}
    $$

    in element-wise notation, where $\mathbf{W}$ is a tensor of shape $(K_k, K_j)$,
    where we have that $K_o = K_qK_k$ is the number of output units.
    Finally, it returns the output tensor of shape $(B, K_o)$ obtained by flattening the
    last two dimensions of the tensor $\mathcal{S}$. Note that the above operations are
    parallelized w.r.t. the additional fold dimension.
    """

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        *,
        weight: TorchParameter,
        semiring: Semiring | None = None,
        num_folds: int = 1,
    ):
        """Initialize a tensor dot layer.

        Args:
            num_input_units: The number of input units $K_i$, such that
                $K_i = K_j K_q$ for some $K_j,K_q$.
            num_output_units: The number of output units $K_o$, such that
                $K_o = K_q K_k$ for some $K_k$.
            weight: The weight parameter, which must have shape $(F, K_k, K_j)$,
                where $F$ is the number of folds, and $K_k,K_j$ are defined
                as in the definition of the number of input and output units above.

        ValueError: If the number of input and output units are incompatible with the
                shape of the weight parameter.
        """
        super().__init__(
            num_input_units,
            num_output_units,
            arity=1,
            semiring=semiring,
            num_folds=num_folds,
        )
        if not self._valid_weight_shape(weight):
            raise ValueError(
                f"Expected number of folds {self.num_folds} "
                f"and shape (K_k, K_j) for 'weight', where "
                f"{self.num_input_units} = K_jK_q and "
                f"{self.num_output_units} = K_qK_k, "
                f"but found {weight.num_folds} and {weight.shape}, respectively"
            )
        self.weight = weight
        self._num_contract_units = weight.shape[1]
        self._num_batch_units = num_input_units // self._num_contract_units

    def _valid_weight_shape(self, w: TorchParameter) -> bool:
        if w.num_folds != self.num_folds:
            return False
        if len(w.shape) != 2:
            return False
        if self.num_input_units % w.shape[1] != 0:
            return False
        if self.num_output_units != w.shape[0] * (self.num_input_units // w.shape[1]):
            return False
        return True

    @property
    def config(self) -> Mapping[str, Any]:
        return {"num_input_units": self.num_input_units, "num_output_units": self.num_output_units}

    @property
    def params(self) -> Mapping[str, TorchParameter]:
        return {"weight": self.weight}

    def forward(self, x: Tensor) -> Tensor:
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
        # return y: (F, B, Kq * Kk) = (F, B, Ko)
        return y.view(y.shape[0], y.shape[1], self.num_output_units)
