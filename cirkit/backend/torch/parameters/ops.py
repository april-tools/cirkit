from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from cirkit.backend.torch.parameters.parameter import TorchParameterOp


class TorchUnaryOpParameter(TorchParameterOp, ABC):
    def __init__(self, in_shape: Tuple[int, ...], num_folds: int = 1) -> None:
        super().__init__(in_shape, num_folds=num_folds)

    def __call__(self, x: Tensor) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(x)  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...


class TorchBinaryOpParameter(TorchParameterOp, ABC):
    def __init__(
        self, in_shape1: Tuple[int, ...], in_shape2: Tuple[int, ...], num_folds: int = 1
    ) -> None:
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)

    def __call__(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(x1, x2)  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        ...


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
        return dict(dim=self.dim)


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
        return dict(dim=self.dim)

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
        return dict(dim=self.dim)

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

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * (self.vmax - self.vmin) + self.vmin


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
