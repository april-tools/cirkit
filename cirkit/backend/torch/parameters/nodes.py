from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, final

import numpy as np
import torch
from torch import Tensor, nn

from cirkit.backend.torch.graph.modules import AbstractTorchModule


class TorchParameterNode(AbstractTorchModule, ABC):
    """The abstract base class for all reparameterizations."""

    def __init__(self, *, num_folds: int = 1, **kwargs) -> None:
        """Init class."""
        super().__init__(num_folds=num_folds)

    @abstractmethod
    def __copy__(self) -> "TorchParameterNode":
        ...

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def config(self) -> Dict[str, Any]:
        return {}

    @torch.no_grad()
    def reset_parameters(self) -> None:
        ...


class TorchParameterInput(TorchParameterNode, ABC):
    @property
    def is_initialized(self) -> bool:
        return True

    def __call__(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self) -> Tensor:
        ...


@final
class TorchTensorParameter(TorchParameterInput):
    """The leaf in reparameterizations that holds the parameter Tensor."""

    def __init__(
        self,
        *shape: int,
        num_folds: int = 1,
        requires_grad: bool = True,
        initializer_: Optional[Callable[[Tensor], Tensor]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Init class."""
        if dtype is None:
            dtype = torch.get_default_dtype()
        super().__init__(num_folds=num_folds)
        self._shape = shape
        self._ptensor: Optional[nn.Parameter] = None
        self._requires_grad = requires_grad
        self._initializer_ = nn.init.normal_ if initializer_ is None else initializer_
        self._dtype = dtype

    def __copy__(self) -> "TorchTensorParameter":
        cls = self.__class__
        return cls(**self.config)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        assert self._ptensor is not None
        return self._ptensor.device

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value
        if self._ptensor is not None:
            self._ptensor.requires_grad = value

    @property
    def initializer(self) -> Callable[[Tensor], Tensor]:
        return self._initializer_

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return self.shape, self.requires_grad, self._dtype

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration flags for the parameter."""
        return dict(
            shape=self._shape,
            num_folds=self.num_folds,
            requires_grad=self._requires_grad,
            initializer_=self._initializer_,
            dtype=self._dtype,
        )

    @property
    def is_initialized(self) -> bool:
        return self._ptensor is not None

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize the internal parameter tensor with the given initializer."""
        if self._ptensor is None:
            shape = (self.num_folds, *self._shape)
            self._ptensor = nn.Parameter(
                torch.empty(*shape, dtype=self._dtype), requires_grad=self._requires_grad
            )
            self._initializer_(self._ptensor.data)
            return
        self._initializer_(self._ptensor.data)

    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        assert self._ptensor is not None
        return self._ptensor


@final
class TorchPointerParameter(TorchParameterInput):
    def __init__(
        self, parameter: TorchTensorParameter, *, fold_idx: Optional[Union[int, List[int]]] = None
    ) -> None:
        if fold_idx is None:
            num_folds = parameter.num_folds
        elif isinstance(fold_idx, int):
            assert 0 <= fold_idx < parameter.num_folds
            if fold_idx == 0 and parameter.num_folds == 1:
                fold_idx = None
                num_folds = parameter.num_folds
            else:
                fold_idx = [fold_idx]
                num_folds = 1
        else:
            assert isinstance(fold_idx, list)
            assert all(0 <= i < parameter.num_folds for i in fold_idx)
            if fold_idx == list(range(parameter.num_folds)):
                fold_idx = None
                num_folds = parameter.num_folds
            else:
                num_folds = len(fold_idx)
        assert not isinstance(parameter, TorchPointerParameter)
        super().__init__(num_folds=num_folds)
        super().__setattr__("_parameter", parameter)
        self.register_buffer("_fold_idx", None if fold_idx is None else torch.tensor(fold_idx))

    def __copy__(self) -> "TorchPointerParameter":
        cls = self.__class__
        return cls(self._parameter, **self.config)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the output parameter."""
        return self._parameter.shape

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return self.shape, id(self._parameter)

    @property
    def config(self) -> Dict[str, Any]:
        return dict(fold_idx=self.fold_idx)

    def deref(self) -> TorchTensorParameter:
        return self._parameter

    @property
    def fold_idx(self) -> Optional[List[int]]:
        if self._fold_idx is None:
            return None
        return self._fold_idx.cpu().tolist()

    def forward(self) -> Tensor:
        x = self._parameter()
        if self._fold_idx is None:
            return x
        return x[self._fold_idx]


class TorchParameterOp(TorchParameterNode, ABC):
    def __init__(self, *in_shape: Tuple[int, ...], num_folds: int = 1):
        super().__init__(num_folds=num_folds)
        self.in_shapes = in_shape

    def __copy__(self) -> "TorchParameterOp":
        cls = self.__class__
        return cls(*self.in_shapes, **self.config)

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return self.in_shapes

    def __call__(self, *xs: Tensor) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(*xs)  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self, *xs: Tensor) -> Tensor:
        ...


class TorchUnaryParameterOp(TorchParameterOp, ABC):
    def __init__(self, in_shape: Tuple[int, ...], *, num_folds: int = 1) -> None:
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


class TorchBinaryParameterOp(TorchParameterOp, ABC):
    def __init__(
        self, in_shape1: Tuple[int, ...], in_shape2: Tuple[int, ...], *, num_folds: int = 1
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


class TorchEntrywiseParameterOp(TorchUnaryParameterOp, ABC):
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.in_shapes[0]


class TorchReduceParameterOp(TorchUnaryParameterOp, ABC):
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

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return *super().fold_settings, self.dim


class TorchEntrywiseReduceParameterOp(TorchEntrywiseParameterOp, ABC):
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

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return *super().fold_settings, self.dim


class TorchIndexParameter(TorchUnaryParameterOp):
    def __init__(
        self, in_shape: Tuple[int, ...], *, num_folds: int = 1, indices: List[int], dim: int = -1
    ) -> None:
        super().__init__(in_shape, num_folds=num_folds)
        dim = dim if dim >= 0 else dim + len(in_shape)
        assert 0 <= dim < len(in_shape)
        assert all(0 <= i < in_shape[dim] for i in indices)
        super().__init__(in_shape, num_folds=num_folds)
        self.dim = dim
        self.register_buffer("_indices", torch.tensor(indices))

    @property
    def indices(self) -> List[int]:
        return self._indices.cpu().tolist()

    @property
    def config(self) -> Dict[str, Any]:
        config = super().config
        config.update(indices=self.indices, dim=self.dim)
        return config

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return *super().fold_settings, tuple(self.indices), self.dim

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            *self.in_shapes[0][: self.dim],
            len(self._indices),
            *self.in_shapes[0][self.dim + 1 :],
        )

    def forward(self, x: Tensor) -> Tensor:
        return x[:, self._indices]


class TorchSumParameter(TorchBinaryParameterOp):
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


class TorchHadamardParameter(TorchBinaryParameterOp):
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


class TorchKroneckerParameter(TorchBinaryParameterOp):
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


class TorchOuterProductParameter(TorchBinaryParameterOp):
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

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return *super().fold_settings, self.dim

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # x1: (F, d1, d2, ..., dk1, ... dn)
        # x2: (F, d1, d2, ..., dk2, ... dn)
        x1 = x1.unsqueeze(self.dim + 2)  # (F, d1, d2, ..., dk1, 1, ..., dn)
        x2 = x2.unsqueeze(self.dim + 1)  # (F, d1, d2, ..., 1, dk1, ...., dn)
        x = x1 * x2  # (F, d1, d2, ..., dk1, dk2, ..., dn)
        x = x.view(self.num_folds, *self.shape)  # (F, d1, d2, ..., dk1 * dk2, ..., dn)
        return x


class TorchOuterSumParameter(TorchBinaryParameterOp):
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

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return *super().fold_settings, self.dim

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # x1: (F, d1, d2, ..., dk1, ... dn)
        # x2: (F, d1, d2, ..., dk2, ... dn)
        x1 = x1.unsqueeze(self.dim + 2)  # (F, d1, d2, ..., dk1, 1, ..., dn)
        x2 = x2.unsqueeze(self.dim + 1)  # (F, d1, d2, ..., 1, dk1, ...., dn)
        x = x1 + x2  # (F, d1, d2, ..., dk1, dk2, ..., dn)
        x = x.view(self.num_folds, *self.shape)  # (F, d1, d2, ..., dk1 * dk2, ..., dn)
        return x


class TorchExpParameter(TorchEntrywiseParameterOp):
    """Exp reparameterization."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(x)


class TorchLogParameter(TorchEntrywiseParameterOp):
    """Log reparameterization."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x)


class TorchSquareParameter(TorchEntrywiseParameterOp):
    """Square reparameterization."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.square(x)


class TorchSigmoidParameter(TorchEntrywiseParameterOp):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x)


class TorchScaledSigmoidParameter(TorchEntrywiseParameterOp):
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

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return *super().fold_settings, self.vmin, self.vmax

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * (self.vmax - self.vmin) + self.vmin


class TorchClampParameter(TorchEntrywiseParameterOp):
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

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return *super().fold_settings, self.vmin, self.vmax

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, min=self.vmin, max=self.vmax)


class TorchConjugateParameter(TorchEntrywiseParameterOp):
    """Conjugate parameterization."""

    def __init__(self, in_shape: Tuple[int, ...], num_folds: int = 1) -> None:
        super().__init__(in_shape, num_folds=num_folds)

    def forward(self, x: Tensor) -> Tensor:
        return torch.conj(x)


class TorchReduceSumParameter(TorchReduceParameterOp):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sum(x, dim=self.dim + 1)


class TorchReduceProductParameter(TorchReduceParameterOp):
    def forward(self, x: Tensor) -> Tensor:
        return torch.prod(x, dim=self.dim + 1)


class TorchReduceLSEParameter(TorchReduceParameterOp):
    def forward(self, x: Tensor) -> Tensor:
        return torch.logsumexp(x, dim=self.dim + 1)


class TorchSoftmaxParameter(TorchEntrywiseReduceParameterOp):
    """Softmax reparameterization.

    Range: (0, 1), 0 available if input is masked, 1 available when only one element valid.
    Constraints: sum to 1.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.softmax(x, dim=self.dim + 1)


class TorchLogSoftmaxParameter(TorchEntrywiseReduceParameterOp):
    """Log-Softmax reparameterization.

    Range: (-inf, 0).
    Constraints: logsumexp is 0.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.log_softmax(x, dim=self.dim + 1)


class TorchMatMulParameter(TorchBinaryParameterOp):
    def __init__(
        self, in_shape1: Tuple[int, ...], in_shape2: Tuple[int, ...], *, num_folds: int = 1
    ) -> None:
        assert len(in_shape1) == len(in_shape2) == 2
        assert in_shape1[1] == in_shape2[0]
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.in_shapes[0][0], self.in_shapes[1][1]

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # x1: (F, d1, d2)
        # x2: (F, d2, d3)
        return torch.matmul(x1, x2)  # (F, d1, d3)


class TorchGaussianProductMean(TorchParameterOp):
    def __init__(
        self,
        in_gaussian1_shape: Tuple[int, ...],
        in_gaussian2_shape: Tuple[int, ...],
        *,
        num_folds: int = 1,
    ) -> None:
        assert in_gaussian1_shape[1] == in_gaussian2_shape[1]
        super().__init__(in_gaussian1_shape, in_gaussian2_shape, num_folds=num_folds)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.in_shapes[0][0] * self.in_shapes[1][0],
            self.in_shapes[0][1],
        )

    def forward(self, mean1: Tensor, mean2: Tensor, stddev1: Tensor, stddev2: Tensor) -> Tensor:
        var1 = torch.square(stddev1)  # (F, K1, C)
        var2 = torch.square(stddev2)  # (F, K2, C)
        inv_var12 = torch.reciprocal(
            var1.unsqueeze(dim=2) + var2.unsqueeze(dim=1)
        )  # (F, K1, K2, C)
        wm1 = mean1.unsqueeze(dim=2) * var2.unsqueeze(dim=1)  # (F, K1, K2, C)
        wm2 = mean2.unsqueeze(dim=1) * var1.unsqueeze(dim=2)  # (F, K1, K2, C)
        mean = (wm1 + wm2) * inv_var12  # (F, K1, K2, C)
        return mean.view(-1, *self.shape)  # (F, K1 * K2, C)


class TorchGaussianProductStddev(TorchBinaryParameterOp):
    def __init__(
        self,
        in_gaussian1_shape: Tuple[int, ...],
        in_gaussian2_shape: Tuple[int, ...],
        *,
        num_folds: int = 1,
    ) -> None:
        assert in_gaussian1_shape[1] == in_gaussian2_shape[1]
        super().__init__(in_gaussian1_shape, in_gaussian2_shape, num_folds=num_folds)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.in_shapes[0][0] * self.in_shapes[1][0],
            self.in_shapes[0][1],
        )

    def forward(self, stddev1: Tensor, stddev2: Tensor) -> Tensor:
        var1 = torch.square(stddev1)  # (F, K1, C)
        var2 = torch.square(stddev2)  # (F, K2, C)
        inv_var1 = torch.reciprocal(var1).unsqueeze(dim=2)  # (F, K1, 1, C)
        inv_var2 = torch.reciprocal(var2).unsqueeze(dim=1)  # (F, 1, K2, C)
        var = torch.reciprocal(inv_var1 + inv_var2)  # (F, K1, K2, C)
        return torch.sqrt(var).view(-1, *self.shape)  # (F, K1 * K2, C)


class TorchGaussianProductLogPartition(TorchParameterOp):
    def __init__(
        self,
        in_gaussian1_shape: Tuple[int, ...],
        in_gaussian2_shape: Tuple[int, ...],
        *,
        num_folds: int = 1,
    ) -> None:
        assert in_gaussian1_shape[1] == in_gaussian2_shape[1]
        super().__init__(in_gaussian1_shape, in_gaussian2_shape, num_folds=num_folds)
        self._log_two_pi = np.log(2.0 * np.pi)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.in_shapes[0][0] * self.in_shapes[1][0],
            self.in_shapes[0][1],
        )

    def forward(
        self,
        mean1: Tensor,
        mean2: Tensor,
        stddev1: Tensor,
        stddev2: Tensor,
    ) -> Tensor:
        var1 = torch.square(stddev1)  # (F, K1, C)
        var2 = torch.square(stddev2)  # (F, K2, C)
        var12 = var1.unsqueeze(dim=2) + var2.unsqueeze(dim=1)  # (F, K1, K2, C)
        inv_var12 = torch.reciprocal(var12)
        sq_mahalanobis = torch.square(mean1.unsqueeze(dim=2) - mean2.unsqueeze(dim=1)) * inv_var12
        log_partition = -0.5 * (self._log_two_pi + torch.log(var12) + sq_mahalanobis)
        return log_partition.view(-1, *self.shape)  # (F, K1 * K2, C)
