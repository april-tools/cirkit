from abc import ABC
from functools import cached_property
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from cirkit.backend.torch.parameters.parameter import (
    TorchBinaryOpParameter,
    TorchParameterOp,
    TorchUnaryOpParameter,
)


class TorchEntrywiseOpParameter(TorchUnaryOpParameter, ABC):
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.in_shapes[0]


class TorchReduceOpParamter(TorchUnaryOpParameter, ABC):
    """The base class for normalized reparameterization."""

    # NOTE: This class only serves as the common base of all normalized reparams, but include
    #       nothing more. It's up to the implementations to define further details.
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        *,
        dim: int = -1,
        num_folds: int = 1,
    ) -> None:
        dim = dim if dim >= 0 else dim + len(in_shape)
        assert 0 <= dim < len(in_shape)
        super().__init__(in_shape, num_folds=num_folds)
        self.dim = dim

    @property
    def shape(self) -> Tuple[int, ...]:
        return *self.in_shapes[0][: self.dim], *self.in_shapes[0][self.dim + 1 :]

    @property
    def config(self) -> Dict[str, Any]:
        return dict(dim=self.dim)


class TorchEntrywiseReduceOpParameter(TorchEntrywiseOpParameter, ABC):
    """The base class for normalized reparameterization."""

    # NOTE: This class only serves as the common base of all normalized reparams, but include
    #       nothing more. It's up to the implementations to define further details.
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        *,
        num_folds: int = 1,
        dim: int = -1,
    ) -> None:
        dim = dim if dim >= 0 else dim + len(in_shape)
        assert 0 <= dim < len(in_shape)
        super().__init__(in_shape, num_folds=num_folds)
        self.dim = dim

    @property
    def config(self) -> Dict[str, Any]:
        config = super().config
        config.update(dim=self.dim)
        return config


class TorchSumParameter(TorchBinaryOpParameter):
    def __init__(
        self, in_shape1: Tuple[int, ...], in_shape2: Tuple[int, ...], *, num_folds: int = 1
    ) -> None:
        assert in_shape1 == in_shape2
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.in_shapes[0]

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return x1 + x2


class TorchHadamardParameter(TorchBinaryOpParameter):
    """Hadamard product reparameterization."""

    def __init__(
        self, in_shape1: Tuple[int, ...], in_shape2: Tuple[int, ...], *, num_folds: int = 1
    ) -> None:
        assert in_shape1 == in_shape2
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.in_shapes[0]

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return x1 * x2


class TorchKroneckerParameter(TorchBinaryOpParameter):
    """Kronecker product reparameterization."""

    def __init__(
        self, in_shape1: Tuple[int, ...], in_shape2: Tuple[int, ...], *, num_folds: int = 1
    ) -> None:
        assert len(in_shape1) == len(in_shape2)
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)
        self._batched_kron = torch.vmap(torch.kron)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(d1 * d2 for d1, d2 in zip(self.in_shapes[0], self.in_shapes[1]))

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self._batched_kron(x1, x2)


class TorchOuterProductParameter(TorchBinaryOpParameter):
    def __init__(
        self,
        in_shape1: Tuple[int, ...],
        in_shape2: Tuple[int, ...],
        *,
        num_folds: int = 1,
        dim: int = -1,
    ) -> None:
        assert len(in_shape1) == len(in_shape2)
        dim = dim if dim >= 0 else dim + len(in_shape1)
        assert 0 <= dim < len(in_shape1)
        assert in_shape1[:dim] == in_shape2[:dim]
        assert in_shape1[dim + 1 :] == in_shape2[dim + 1 :]
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)
        self.dim = dim

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            *self.in_shapes[0][: self.dim],
            self.in_shapes[0][self.dim] * self.in_shapes[1][self.dim],
            *self.in_shapes[0][self.dim + 1 :],
        )

    @property
    def config(self) -> Dict[str, Any]:
        config = super().config
        config.update(dim=self.dim)
        return config

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # x1: (F, d1, d2, ..., dk1, ... dn)
        # x2: (F, d1, d2, ..., dk2, ... dn)
        x1 = x1.unsqueeze(self.dim + 2)  # (F, d1, d2, ..., dk1, 1, ..., dn)
        x2 = x2.unsqueeze(self.dim + 1)  # (F, d1, d2, ..., 1, dk1, ...., dn)
        x = x1 * x2  # (F, d1, d2, ..., dk1, dk2, ..., dn)
        x = x.view(self.num_folds, *self.shape)  # (F, d1, d2, ..., dk1 * dk2, ..., dn)
        return x


class TorchOuterSumParameter(TorchBinaryOpParameter):
    def __init__(
        self,
        in_shape1: Tuple[int, ...],
        in_shape2: Tuple[int, ...],
        *,
        num_folds: int = 1,
        dim: int = -1,
    ) -> None:
        assert len(in_shape1) == len(in_shape2)
        dim = dim if dim >= 0 else dim + len(in_shape1)
        assert 0 <= dim < len(in_shape1)
        assert in_shape1[:dim] == in_shape2[:dim]
        assert in_shape1[dim + 1 :] == in_shape2[dim + 1 :]
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)
        self.dim = dim

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            *self.in_shapes[0][: self.dim],
            self.in_shapes[0][self.dim] * self.in_shapes[1][self.dim],
            *self.in_shapes[0][self.dim + 1 :],
        )

    @property
    def config(self) -> Dict[str, Any]:
        config = super().config
        config.update(dim=self.dim)
        return config

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # x1: (F, d1, d2, ..., dk1, ... dn)
        # x2: (F, d1, d2, ..., dk2, ... dn)
        x1 = x1.unsqueeze(self.dim + 2)  # (F, d1, d2, ..., dk1, 1, ..., dn)
        x2 = x2.unsqueeze(self.dim + 1)  # (F, d1, d2, ..., 1, dk1, ...., dn)
        x = x1 + x2  # (F, d1, d2, ..., dk1, dk2, ..., dn)
        x = x.view(self.num_folds, *self.shape)  # (F, d1, d2, ..., dk1 * dk2, ..., dn)
        return x


class TorchExpParameter(TorchEntrywiseOpParameter):
    """Exp reparameterization."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(x)


class TorchLogParameter(TorchEntrywiseOpParameter):
    """Log reparameterization."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x)


class TorchSquareParameter(TorchEntrywiseOpParameter):
    """Square reparameterization."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.square(x)


class TorchSigmoidParameter(TorchEntrywiseOpParameter):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x)


class TorchScaledSigmoidParameter(TorchEntrywiseOpParameter):
    def __init__(
        self, in_shape: Tuple[int, ...], *, vmin: float, vmax: float, num_folds: int = 1
    ) -> None:
        super().__init__(in_shape, num_folds=num_folds)
        assert 0 <= vmin < vmax, "Must provide 0 <= vmin < vmax."
        self.vmin = vmin
        self.vmax = vmax

    @property
    def config(self) -> Dict[str, Any]:
        config = super().config
        config.update(vmin=self.vmin, vmax=self.vmax)
        return config

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * (self.vmax - self.vmin) + self.vmin


class TorchClampParameter(TorchEntrywiseOpParameter):
    """Exp reparameterization."""

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        num_folds: int = 1,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        assert vmin is not None or vmax is not None
        super().__init__(in_shape, num_folds=num_folds)
        self.vmin = vmin
        self.vmax = vmax

    @property
    def config(self) -> Dict[str, Any]:
        config = super().config
        if self.vmin is not None:
            config.update(vmin=self.vmin)
        if self.vmax is not None:
            config.update(vmax=self.vmax)
        return config

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, min=self.vmin, max=self.vmax)


class TorchReduceSumParameter(TorchReduceOpParamter):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sum(x, dim=self.dim + 1)


class TorchReduceProductParameter(TorchReduceOpParamter):
    def forward(self, x: Tensor) -> Tensor:
        return torch.prod(x, dim=self.dim + 1)


class TorchReduceLSEParameter(TorchReduceOpParamter):
    def forward(self, x: Tensor) -> Tensor:
        return torch.logsumexp(x, dim=self.dim + 1)


class TorchSoftmaxParameter(TorchEntrywiseReduceOpParameter):
    """Softmax reparameterization.

    Range: (0, 1), 0 available if input is masked, 1 available when only one element valid.
    Constraints: sum to 1.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.softmax(x, dim=self.dim + 1)


class TorchLogSoftmaxParameter(TorchEntrywiseReduceOpParameter):
    """Log-Softmax reparameterization.

    Range: (-inf, 0).
    Constraints: logsumexp is 0.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.log_softmax(x, dim=self.dim + 1)


class TorchGaussianProductMean(TorchParameterOp):
    def __init__(
        self,
        in_gaussian1_shape: Tuple[int, ...],
        in_gaussian2_shape: Tuple[int, ...],
        *,
        num_folds: int = 1,
    ) -> None:
        assert (
            in_gaussian1_shape[0] == in_gaussian2_shape[0]
            and in_gaussian1_shape[2] == in_gaussian2_shape[2]
        )
        super().__init__(in_gaussian1_shape, in_gaussian2_shape, num_folds=num_folds)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.in_shapes[0][0],
            self.in_shapes[0][1] * self.in_shapes[1][1],
            self.in_shapes[0][2],
        )

    def forward(self, mean1: Tensor, mean2: Tensor, stddev1: Tensor, stddev2: Tensor) -> Tensor:
        var1 = torch.square(stddev1)  # (F, D, K1, C)
        var2 = torch.square(stddev2)  # (F, D, K2, C)
        inv_var12 = torch.reciprocal(
            var1.unsqueeze(dim=3) + var2.unsqueeze(dim=2)
        )  # (F, D, K1, K2, C)
        wm1 = mean1.unsqueeze(dim=3) * var2.unsqueeze(dim=2)  # (F, D, K1, K2, C)
        wm2 = mean2.unsqueeze(dim=2) * var1.unsqueeze(dim=3)  # (F, D, K1, K2, C)
        mean = (wm1 + wm2) * inv_var12  # (F, D, K1, K2, C)
        return mean.view(-1, *self.shape)  # (F, D, K1 * K2, C)


class TorchGaussianProductStddev(TorchBinaryOpParameter):
    def __init__(
        self,
        in_gaussian1_shape: Tuple[int, ...],
        in_gaussian2_shape: Tuple[int, ...],
        *,
        num_folds: int = 1,
    ) -> None:
        assert (
            in_gaussian1_shape[0] == in_gaussian2_shape[0]
            and in_gaussian1_shape[2] == in_gaussian2_shape[2]
        )
        super().__init__(in_gaussian1_shape, in_gaussian2_shape, num_folds=num_folds)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.in_shapes[0][0],
            self.in_shapes[0][1] * self.in_shapes[1][1],
            self.in_shapes[0][2],
        )

    def forward(self, stddev1: Tensor, stddev2: Tensor) -> Tensor:
        var1 = torch.square(stddev1)  # (F, D, K1, C)
        var2 = torch.square(stddev2)  # (F, D, K2, C)
        inv_var1 = torch.reciprocal(var1).unsqueeze(dim=3)  # (F, D, K1, 1, C)
        inv_var2 = torch.reciprocal(var2).unsqueeze(dim=2)  # (F, D, 1, K2, C)
        var = torch.reciprocal(inv_var1 + inv_var2)  # (F, D, K1, K2, C)
        return torch.sqrt(var).view(-1, *self.shape)  # (F, D, K1 * K2print, C)


class TorchGaussianProductLogPartition(TorchParameterOp):
    def __init__(
        self,
        in_gaussian1_shape: Tuple[int, ...],
        in_gaussian2_shape: Tuple[int, ...],
        *,
        num_folds: int = 1,
    ) -> None:
        assert (
            in_gaussian1_shape[0] == in_gaussian2_shape[0]
            and in_gaussian1_shape[2] == in_gaussian2_shape[2]
        )
        super().__init__(in_gaussian1_shape, in_gaussian2_shape, num_folds=num_folds)
        self._log_two_pi = np.log(2.0 * np.pi)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.in_shapes[0][0],
            self.in_shapes[0][1] * self.in_shapes[1][1],
            self.in_shapes[0][2],
        )

    def forward(
        self,
        mean1: Tensor,
        mean2: Tensor,
        stddev1: Tensor,
        stddev2: Tensor,
    ) -> Tensor:
        var1 = torch.square(stddev1)  # (F, D, K1, C)
        var2 = torch.square(stddev2)  # (F, D, K1, C)
        var12 = var1.unsqueeze(dim=3) + var2.unsqueeze(dim=2)  # (F, D, K1, K2, C)
        inv_var12 = torch.reciprocal(var12)
        sq_mahalanobis = torch.square(mean1.unsqueeze(dim=3) - mean2.unsqueeze(dim=2)) * inv_var12
        log_partition = -0.5 * (self._log_two_pi + torch.log(var12) + sq_mahalanobis)
        return log_partition.view(-1, *self.shape)  # (F, D, K1 * K2, C)
