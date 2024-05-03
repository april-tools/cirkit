from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from cirkit.backend.torch.params.base import AbstractTorchParameter


class TorchOpParameter(AbstractTorchParameter, ABC):
    def __init__(self, *, num_folds: int = 1):
        super().__init__(num_folds=num_folds)

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """The shape of the output parameter."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The device of the output parameter."""


class TorchUnaryOpParameter(TorchOpParameter, ABC):
    def __init__(self, opd: AbstractTorchParameter) -> None:
        super().__init__(num_folds=opd.num_folds)
        self.opd = opd

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""
        return self.opd.dtype

    @property
    def device(self) -> torch.device:
        """The device of the output parameter."""
        return self.opd.device

    @property
    def params(self) -> Dict[str, 'AbstractTorchParameter']:
        """The other parameters this parameter depends on."""
        return dict(opd=self.opd)

    @abstractmethod
    def _forward_impl(self, x: Tensor) -> Tensor:
        ...

    def forward(self) -> Tensor:
        return self._forward_impl(self.opd())


class TorchEntrywiseOpParameter(TorchUnaryOpParameter, ABC):
    def __init__(self, opd: AbstractTorchParameter):
        super().__init__(opd)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.opd.shape


class TorchEntrywiseReduceOpParameter(TorchEntrywiseOpParameter, ABC):
    """The base class for normalized reparameterization."""

    # NOTE: This class only serves as the common base of all normalized reparams, but include
    #       nothing more. It's up to the implementations to define further details.
    def __init__(
        self,
        opd: AbstractTorchParameter,
        *,
        dim: int = -1,
    ) -> None:
        super().__init__(opd)
        dim = dim if dim >= 0 else dim + len(opd.shape)
        assert 0 <= dim < len(opd.shape)
        self.dim = dim

    @property
    def config(self) -> Dict[str, Any]:
        return dict(dim=self.dim)


class TorchReduceOpParamter(TorchUnaryOpParameter, ABC):
    """The base class for normalized reparameterization."""

    # NOTE: This class only serves as the common base of all normalized reparams, but include
    #       nothing more. It's up to the implementations to define further details.
    def __init__(
        self,
        opd: AbstractTorchParameter,
        *,
        dim: int = -1,
    ) -> None:
        super().__init__(opd)
        dim = dim if dim >= 0 else dim + len(opd.shape)
        assert 0 <= dim < len(opd.shape)
        self.dim = dim

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return *self.opd.shape[: self.dim], *self.opd.shape[self.dim + 1 :]

    @property
    def config(self) -> Dict[str, Any]:
        return dict(dim=self.dim)


class TorchBinaryOpParameter(TorchOpParameter, ABC):
    def __init__(self, opd1: AbstractTorchParameter, opd2: AbstractTorchParameter) -> None:
        assert opd1.num_folds == opd2.num_folds
        super().__init__(num_folds=opd1.num_folds)
        self.opd1 = opd1
        self.opd2 = opd2

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""
        assert self.opd1.dtype == self.opd2.dtype
        return self.opd1.dtype

    @property
    def device(self) -> torch.device:
        """The device of the output parameter."""
        assert self.opd1.dtype == self.opd2.dtype
        return self.opd1.device

    @property
    def params(self) -> Dict[str, 'AbstractTorchParameter']:
        """The other parameters this parameter depends on."""
        return dict(opd1=self.opd1, opd2=self.opd2)

    @abstractmethod
    def _forward_impl(self, x1: Tensor, x2: Tensor) -> Tensor:
        ...

    def forward(self) -> Tensor:
        return self._forward_impl(self.opd1(), self.opd2())


class TorchOuterSumParameter(TorchBinaryOpParameter):
    def __init__(
        self, opd1: AbstractTorchParameter, opd2: AbstractTorchParameter, *, dim: int = -1
    ) -> None:
        assert opd1.num_folds == opd2.num_folds and len(opd1.shape) == len(opd2.shape)
        dim = dim if dim >= 0 else dim + len(opd1.shape)
        assert 0 <= dim < len(opd1.shape)
        assert opd1.shape[:dim] == opd2.shape[:dim] and opd1.shape[dim + 1:] == opd2.shape[dim + 1:]
        super().__init__(opd1, opd2)
        self.dim = dim

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return (
            *self.opd1.shape[: self.dim],
            self.opd1.shape[self.dim] * self.opd2.shape[self.dim],
            *self.opd1.shape[self.dim + 1 :],
        )

    @property
    def config(self) -> Dict[str, Any]:
        return dict(dim=self.dim)

    def _forward_impl(self, x1: Tensor, x2: Tensor) -> Tensor:
        # x1: (F, d1, d2, ..., dk1, ... dn)
        # x2: (F, d1, d2, ..., dk2, ... dn)
        x1 = x1.unsqueeze(self.dim + 2)  # (F, d1, d2, ..., dk1, 1, ..., dn)
        x2 = x2.unsqueeze(self.dim + 1)  # (F, d1, d2, ..., 1, dk1, ...., dn)
        x = x1 + x2                      # (F, d1, d2, ..., dk1, dk2, ..., dn)
        x = x.view(self.num_folds, *self.shape)  # (F, d1, d2, ..., dk1 * dk2, ..., dn)
        return x


class TorchHadamardParameter(TorchBinaryOpParameter):
    """Hadamard product reparameterization."""

    def __init__(self, opd1: AbstractTorchParameter, opd2: AbstractTorchParameter) -> None:
        assert opd1.num_folds == opd2.num_folds and opd1.shape == opd2.shape
        super().__init__(opd1, opd2)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return self.opd1.shape

    def _forward_impl(self, x1: Tensor, x2: Tensor) -> Tensor:
        return x1 * x2


class TorchKroneckerParameter(TorchBinaryOpParameter):
    """Kronecker product reparameterization."""

    def __init__(self, opd1: AbstractTorchParameter, opd2: AbstractTorchParameter) -> None:
        assert opd1.num_folds == opd2.num_folds and len(opd1.shape) == len(opd2.shape)
        super().__init__(opd1, opd2)
        self._batched_kron = torch.vmap(torch.kron)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(
            self.opd1.shape[i] * self.opd2.shape[i]
            for i in range(len(self.opd1.shape))
        )

    def _forward_impl(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self._batched_kron(x1, x2)


class TorchExpParameter(TorchEntrywiseOpParameter):
    """Exp reparameterization."""

    def __init__(self, opd: AbstractTorchParameter) -> None:
        super().__init__(opd)

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.exp(x)


class TorchLogParameter(TorchEntrywiseOpParameter):
    """Log reparameterization."""

    def __init__(self, opd: AbstractTorchParameter) -> None:
        super().__init__(opd)

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.log(x)


class TorchSquareParameter(TorchEntrywiseOpParameter):
    """Square reparameterization."""

    def __init__(self, opd: AbstractTorchParameter) -> None:
        super().__init__(opd)

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.square(x)


class TorchScaledSigmoidParameter(TorchEntrywiseOpParameter):
    def __init__(
        self,
        opd: AbstractTorchParameter,
        *,
        vmin: float = 0.0,
        vmax: float = 1.0,
    ) -> None:
        """Init class.

        Args:
            opd (Optional[Reparameterization], optional): The input reparam to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
            vmin (float, optional): The minimum value. Defaults to 0.0.
            vmax (float, optional): The maximum value. Defaults to 1.0.
        """
        super().__init__(opd)
        assert 0 <= vmin < vmax, "Must provide 0 <= vmin < vmax."
        self.vmin = vmin
        self.vmax = vmax

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * (self.vmax - self.vmin) + self.vmin


class TorchReduceSumParameter(TorchReduceOpParamter):
    def __init__(self, opd: AbstractTorchParameter, *, dim: int = -1) -> None:
        super().__init__(opd, dim=dim)

    @property
    def config(self) -> Dict[str, Any]:
        return dict(dim=self.dim)

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.sum(x, dim=self.dim + 1)


class TorchReduceLSEParameter(TorchReduceOpParamter):
    def __init__(self, opd: AbstractTorchParameter, *, dim: int = -1) -> None:
        super().__init__(opd, dim=dim)

    @property
    def config(self) -> Dict[str, Any]:
        return dict(dim=self.dim)

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.logsumexp(x, dim=self.dim + 1)


class TorchSoftmaxParameter(TorchEntrywiseReduceOpParameter):
    """Softmax reparameterization.

    Range: (0, 1), 0 available if input is masked, 1 available when only one element valid.
    Constraints: sum to 1.
    """

    def __init__(self, opd: AbstractTorchParameter, *, dim: int = -1) -> None:
        """Init class.

        Args:
            opd (Optional[Reparameterization], optional): The input param to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
        """
        super().__init__(opd, dim=dim)

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.softmax(x, dim=self.dim + 1)


class TorchLogSoftmaxParameter(TorchEntrywiseReduceOpParameter):
    """Log-Softmax reparameterization.

    Range: (-inf, 0).
    Constraints: logsumexp is 0.
    """

    def __init__(self, opd: AbstractTorchParameter, *, dim: int = -1) -> None:
        """Init class.

        Args:
            opd (Optional[Reparameterization], optional): The input param to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
        """
        super().__init__(opd, dim=dim)

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.log_softmax(x, dim=self.dim + 1)
