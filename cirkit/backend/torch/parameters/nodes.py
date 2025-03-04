from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Any, final

import numpy as np
import torch
from torch import Tensor, nn

from cirkit.backend.torch.graph.modules import AbstractTorchModule
from cirkit.backend.torch.utils import CachedGateFunctionEval


class TorchParameterNode(AbstractTorchModule, ABC):
    """The abstract parameter node class. A parameter node is a node in the computational
    graph that computes parameters.
    See [TorchParameter][cirkit.backend.torch.parameters.parameter.TorchParameter]
    for more details."""

    def __init__(self, *, num_folds: int = 1):
        """Initialize a torch parameter node.

        Args:
            num_folds: The number of folds computed by the node.
        """
        super().__init__(num_folds=num_folds)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        r"""The shape of the tensor folds that the node outputs.
        If the shape is $(K_1,\ldots,K_n)$ and the number of folds is $F$, then the node outputs a
        tensor having overall shape $(F,K_1,\ldots,K_n)$.

        Returns:
            The shape of the thensor folds that the node outputs.
        """

    @property
    def config(self) -> dict[str, Any]:
        """Retrieves the configuration of the parameter node, i.e., a dictionary mapping
        hyperparameters of the parameter node to their values. The hyperparameter names must
        match the argument names in the ```__init__``` method.

        Returns:
            Dict[str, Any]: A dictionary from hyperparameter names to their value.
        """
        return {}

    @property
    def fold_settings(self) -> tuple[Any, ...]:
        return (*self.config.items(),)

    @final
    @property
    def sub_modules(self) -> dict[str, "AbstractTorchModule"]:
        return {}

    @torch.no_grad()
    def reset_parameters(self):
        ...


class TorchParameterInput(TorchParameterNode, ABC):
    """The torch parameter input node. A parameter input is a parameter node in the
    computational graph that comptues parameter that does __not__ have inputs. See
    [TorchParameter][cirkit.backend.torch.parameters.parameter.TorchParameter] for more details.
    """

    def __call__(self) -> Tensor:
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    def extra_repr(self) -> str:
        return f"output-shape: {(self.num_folds, *self.shape)}"

    @abstractmethod
    def forward(self) -> Tensor:
        r"""Evaluate a torch parameter input node.

        Returns:
            Tensor: A tensor of shape $(F,K_1,\ldots,K_n)$, where $F$ is the number of folds, and
            $(K_1,\ldots,K_n)$ is the shape of the tensors within each fold.
        """


class TorchTensorParameter(TorchParameterInput):
    """A torch tensor parameter is a
    [TorchParameterInput][cirkit.backend.torch.parameters.nodes.TorchParameterInput]
    that stores a [torch.nn.parameter.Parameter][torch.nn.parameter.Parameter] object.
    """

    def __init__(
        self,
        *shape: int,
        requires_grad: bool = True,
        dtype: torch.dtype | None = None,
        initializer_: Callable[[Tensor], Tensor] | None = None,
        num_folds: int = 1,
    ):
        r"""Initializes a torch tensor parameter. Given a shape $(K_1,\ldots,K_n)$ and a number of
        folds $F$, it eventually materializes a torch parameter of shape $(F,K_1,\ldots,K_n)$.

        Args:
            *shape: The shape of the tensor parameter folds $(K_1,\ldots,K_n)$.
            requires_grad: Whether the parameter requires the computation of gradients.
            dtype: The data type of the parameter.
                If it is None, then it defaults to the current default torch data type, i.e.,
                it is given by [torch.get_default_dtype][torch.get_default_dtype].
            initializer_: The in-place initializer used to initialize the tensor parameter.
                It is a callable with only a tensor as input. If it is None, then it defaults to
                sampling from a standard normal distribution, i.e.,
                [torch.nn.init.normal_][torch.nn.init.normal_].
            num_folds: The number of folds $F$.
        """
        if dtype is None:
            dtype = torch.get_default_dtype()
        super().__init__(num_folds=num_folds)
        self._shape = shape
        self._ptensor: nn.Parameter | None = None
        self._requires_grad = requires_grad
        self._dtype = dtype
        self._initializer_ = nn.init.normal_ if initializer_ is None else initializer_

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        """Retrieve the data type of the parameter.

        Returns:
            torch.dtype: The parameter data type.
        """
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Retrieve the device of the parameter.

        Returns:
            torch.device: The parameter device.

        Raises:
            ValueError: If the parameter has not been initialized.
                See the [reset_parameters][cirkit.backend.torch.parameters.nodes.TorchTensorParameter.reset_parameters]
                method.
        """
        if self._ptensor is None:
            raise ValueError(
                "The tensor parameter has not been initialized. " "Use reset_parameters() first"
            )
        return self._ptensor.device

    @property
    def requires_grad(self) -> bool:
        """Retrieve whether the torch parameter requires gradients.

        Returns:
            bool: True if it requires gradients, False otherwise.
        """
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        """Set whether the torch parameter requires gradients.

        Args:
            value: The value to set.
        """
        self._requires_grad = value
        if self._ptensor is not None:
            self._ptensor.requires_grad = value

    @property
    def initializer(self) -> Callable[[Tensor], Tensor]:
        """Retrieve the initializer of the torch tensor parameter.

        Returns:
            Callable[[Tensor], Tensor]: The in-place tensor initializer.
        """
        return self._initializer_

    @property
    def config(self) -> dict[str, Any]:
        return {
            "shape": self._shape,
            "requires_grad": self._requires_grad,
            "dtype": self._dtype,
            "initializer_": self._initializer_,
        }

    @property
    def fold_settings(self) -> tuple[Any, ...]:
        return self._shape, self._requires_grad, self._dtype

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Allocate and initialize the torch tensor parameter. If the tensor has already been
        allocated, then this function simply call the initializer to reset the parameter values.
        """
        if self._ptensor is None:
            shape = (self.num_folds, *self._shape)
            self._ptensor = nn.Parameter(
                torch.empty(*shape, dtype=self._dtype), requires_grad=self._requires_grad
            )
            self._initializer_(self._ptensor.data)
            return
        self._initializer_(self._ptensor.data)

    def forward(self) -> Tensor:
        r"""Evaluate a torch parameter input node.

        Returns:
            Tensor: A tensor of shape $(F,K_1,\ldots,K_n)$, where $F$ is the number of folds, and
            $(K_1,\ldots,K_n)$ is the shape of the tensors within each fold.

        Raises:
            ValueError: If the parameter has not been initialized.
                See the [reset_parameters][cirkit.backend.torch.parameters.nodes.TorchTensorParameter.reset_parameters]
                method.
        """
        if self._ptensor is None:
            raise ValueError(
                "The tensor parameter has not been initialized. " "Use reset_parameters() first"
            )
        return self._ptensor


class TorchPointerParameter(TorchParameterInput):
    def __init__(
        self, parameter: TorchTensorParameter, *, fold_idx: int | list[int] | None = None
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
        self._parameter = parameter
        self.register_buffer("_fold_idx", None if fold_idx is None else torch.tensor(fold_idx))

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the output parameter."""
        return self._parameter.shape

    @property
    def fold_idx(self) -> list[int] | None:
        if self._fold_idx is None:
            return None
        return self._fold_idx.cpu().tolist()

    @property
    def config(self) -> dict[str, Any]:
        return {"parameter": self._parameter}

    def deref(self) -> TorchTensorParameter:
        return self._parameter

    def forward(self) -> Tensor:
        x = self._parameter()
        if self._fold_idx is None:
            return x
        return x[self._fold_idx]


class TorchGateFunctionParameter(TorchParameterInput):
    def __init__(
        self,
        *shape: int,
        gate_function_eval: CachedGateFunctionEval,
        name: str,
        fold_idx: int | list[int],
    ):
        fold_idx = fold_idx if isinstance(fold_idx, list) else [fold_idx]
        super().__init__(num_folds=len(fold_idx))
        self._gate_function_eval = gate_function_eval
        self._shape = shape
        self._name = name
        self.register_buffer("_fold_idx", torch.tensor(fold_idx))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def gate_function_eval(self) -> CachedGateFunctionEval:
        return self._gate_function_eval

    @property
    def name(self) -> str:
        return self._name

    @property
    def fold_idx(self) -> list[int]:
        return self._fold_idx.cpu().tolist()

    @property
    def config(self) -> dict[str, Any]:
        return {
            "shape": self._shape,
            "gate_function_eval": self._gate_function_eval,
            "name": self._name,
        }

    def forward(self) -> Tensor:
        # A dictionary from gate functions to their tensor value
        y = self._gate_function_eval()  # shape: (group_size, K_1, ..., K_n)
        # Slice the tensor by using the fold index
        return y[self._fold_idx]  # shape: (F, K_1, ..., K_n)


class TorchParameterOp(TorchParameterNode, ABC):
    def __init__(self, *in_shapes: tuple[int, ...], num_folds: int = 1):
        super().__init__(num_folds=num_folds)
        self._in_shapes = in_shapes

    @property
    def in_shapes(self) -> tuple[tuple[int, ...], ...]:
        return self._in_shapes

    @property
    def config(self) -> dict[str, Any]:
        return {"in_shapes": self.in_shapes}

    def __call__(self, *xs: Tensor) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(*xs)  # type: ignore[no-any-return,misc]

    def extra_repr(self) -> str:
        return (
            f"input-shapes: {[(self.num_folds, *in_shape) for in_shape in self._in_shapes]}"
            + "\n"
            + f"output-shape: {(self.num_folds, *self.shape)}"
        )

    @abstractmethod
    def forward(self, *xs: Tensor) -> Tensor:
        ...


class TorchUnaryParameterOp(TorchParameterOp, ABC):
    def __init__(self, in_shape: tuple[int, ...], *, num_folds: int = 1) -> None:
        super().__init__(in_shape, num_folds=num_folds)

    @property
    def in_shape(self) -> tuple[int, ...]:
        (in_shape,) = self.in_shapes
        return in_shape

    @property
    def config(self) -> dict[str, Any]:
        return {"in_shape": self.in_shape}

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
        self, in_shape1: tuple[int, ...], in_shape2: tuple[int, ...], *, num_folds: int = 1
    ) -> None:
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)

    @property
    def in_shape1(self) -> tuple[int, ...]:
        in_shape, _ = self.in_shapes
        return in_shape

    @property
    def in_shape2(self) -> tuple[int, ...]:
        _, in_shape = self.in_shapes
        return in_shape

    @property
    def config(self) -> dict[str, Any]:
        return {"in_shape1": self.in_shape1, "in_shape2": self.in_shape2}

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
    def shape(self) -> tuple[int, ...]:
        return self.in_shape


class TorchReduceParameterOp(TorchUnaryParameterOp, ABC):
    """The base class for normalized reparameterization."""

    # NOTE: This class only serves as the common base of all normalized reparams, but include
    #       nothing more. It's up to the implementations to define further details.
    def __init__(
        self,
        in_shape: tuple[int, ...],
        dim: int = -1,
        *,
        num_folds: int = 1,
    ) -> None:
        dim = dim if dim >= 0 else dim + len(in_shape)
        assert 0 <= dim < len(in_shape)
        super().__init__(in_shape, num_folds=num_folds)
        self.dim = dim

    @property
    def shape(self) -> tuple[int, ...]:
        return *self.in_shape[: self.dim], *self.in_shape[self.dim + 1 :]

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["dim"] = self.dim
        return config


class TorchEntrywiseReduceParameterOp(TorchEntrywiseParameterOp, ABC):
    """The base class for normalized reparameterization."""

    # NOTE: This class only serves as the common base of all normalized reparams, but include
    #       nothing more. It's up to the implementations to define further details.
    def __init__(
        self,
        in_shape: tuple[int, ...],
        *,
        dim: int = -1,
        num_folds: int = 1,
    ) -> None:
        dim = dim if dim >= 0 else dim + len(in_shape)
        assert 0 <= dim < len(in_shape)
        super().__init__(in_shape, num_folds=num_folds)
        self.dim = dim

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["dim"] = self.dim
        return config


class TorchIndexParameter(TorchUnaryParameterOp):
    def __init__(
        self,
        in_shape: tuple[int, ...],
        indices: list[int],
        dim: int = -1,
        *,
        num_folds: int = 1,
    ) -> None:
        super().__init__(in_shape, num_folds=num_folds)
        dim = dim if dim >= 0 else dim + len(in_shape)
        assert 0 <= dim < len(in_shape)
        assert all(0 <= i < in_shape[dim] for i in indices)
        super().__init__(in_shape, num_folds=num_folds)
        self.dim = dim
        self.register_buffer("_indices", torch.tensor(indices))

    @property
    def indices(self) -> list[int]:
        return self._indices.cpu().tolist()

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["indices"] = self.indices
        config["dim"] = self.dim
        return config

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            *self.in_shape[: self.dim],
            len(self._indices),
            *self.in_shape[self.dim + 1 :],
        )

    def forward(self, x: Tensor) -> Tensor:
        return x[:, self._indices]


class TorchSumParameter(TorchBinaryParameterOp):
    def __init__(
        self, in_shape1: tuple[int, ...], in_shape2: tuple[int, ...], *, num_folds: int = 1
    ) -> None:
        assert in_shape1 == in_shape2
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.in_shape1

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return x1 + x2


class TorchHadamardParameter(TorchBinaryParameterOp):
    """Hadamard product reparameterization."""

    def __init__(
        self, in_shape1: tuple[int, ...], in_shape2: tuple[int, ...], *, num_folds: int = 1
    ) -> None:
        assert in_shape1 == in_shape2
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.in_shape1

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return x1 * x2


class TorchKroneckerParameter(TorchBinaryParameterOp):
    """Kronecker product reparameterization."""

    def __init__(
        self, in_shape1: tuple[int, ...], in_shape2: tuple[int, ...], *, num_folds: int = 1
    ) -> None:
        assert len(in_shape1) == len(in_shape2)
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)
        self._batched_kron = torch.vmap(torch.kron)

    @cached_property
    def shape(self) -> tuple[int, ...]:
        return tuple(d1 * d2 for d1, d2 in zip(self.in_shape1, self.in_shape2))

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self._batched_kron(x1, x2)


class TorchOuterProductParameter(TorchBinaryParameterOp):
    def __init__(
        self,
        in_shape1: tuple[int, ...],
        in_shape2: tuple[int, ...],
        dim: int = -1,
        *,
        num_folds: int = 1,
    ) -> None:
        assert len(in_shape1) == len(in_shape2)
        dim = dim if dim >= 0 else dim + len(in_shape1)
        assert 0 <= dim < len(in_shape1)
        assert in_shape1[:dim] == in_shape2[:dim]
        assert in_shape1[dim + 1 :] == in_shape2[dim + 1 :]
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)
        self.dim = dim

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            *self.in_shape1[: self.dim],
            self.in_shape1[self.dim] * self.in_shape2[self.dim],
            *self.in_shape1[self.dim + 1 :],
        )

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["dim"] = self.dim
        return config

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
        in_shape1: tuple[int, ...],
        in_shape2: tuple[int, ...],
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
    def shape(self) -> tuple[int, ...]:
        return (
            *self.in_shape1[: self.dim],
            self.in_shape1[self.dim] * self.in_shape2[self.dim],
            *self.in_shape1[self.dim + 1 :],
        )

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["dim"] = self.dim
        return config

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
        self, in_shape: tuple[int, ...], vmin: float, vmax: float, *, num_folds: int = 1
    ) -> None:
        super().__init__(in_shape, num_folds=num_folds)
        assert 0 <= vmin < vmax, "Must provide 0 <= vmin < vmax."
        self.vmin = vmin
        self.vmax = vmax

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["vmin"] = self.vmin
        config["vmax"] = self.vmax
        return config

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * (self.vmax - self.vmin) + self.vmin


class TorchClampParameter(TorchEntrywiseParameterOp):
    """Exp reparameterization."""

    def __init__(
        self,
        in_shape: tuple[int, ...],
        vmin: float | None = None,
        vmax: float | None = None,
        *,
        num_folds: int = 1,
    ) -> None:
        assert vmin is not None or vmax is not None
        super().__init__(in_shape, num_folds=num_folds)
        self.vmin = vmin
        self.vmax = vmax

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        if self.vmin is not None:
            config["vmin"] = self.vmin
        if self.vmax is not None:
            config["vmax"] = self.vmax
        return config

    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, min=self.vmin, max=self.vmax)


class TorchConjugateParameter(TorchEntrywiseParameterOp):
    """Conjugate parameterization."""

    def __init__(self, in_shape: tuple[int, ...], *, num_folds: int = 1) -> None:
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
        self, in_shape1: tuple[int, ...], in_shape2: tuple[int, ...], *, num_folds: int = 1
    ) -> None:
        assert len(in_shape1) == len(in_shape2) == 2
        assert in_shape1[1] == in_shape2[0]
        super().__init__(in_shape1, in_shape2, num_folds=num_folds)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.in_shape1[0], self.in_shape2[1]

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # x1: (F, d1, d2)
        # x2: (F, d2, d3)
        return torch.matmul(x1, x2)  # (F, d1, d3)


class TorchFlattenParameter(TorchUnaryParameterOp):
    def __init__(
        self,
        in_shape: tuple[int, ...],
        num_folds: int = 1,
        start_dim: int = 0,
        end_dim: int = -1,
    ):
        super().__init__(in_shape, num_folds=num_folds)
        start_dim = start_dim if start_dim >= 0 else start_dim + len(in_shape)
        assert 0 <= start_dim < len(in_shape)
        end_dim = end_dim if end_dim >= 0 else end_dim + len(in_shape)
        assert 0 <= end_dim < len(in_shape)
        assert start_dim < end_dim
        self.start_dim = start_dim
        self.end_dim = end_dim

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["start_dim"] = self.start_dim
        config["end_dim"] = self.end_dim
        return config

    @cached_property
    def shape(self) -> tuple[int, ...]:
        flattened_dim = np.prod(
            [self.in_shapes[0][i] for i in range(self.start_dim, self.end_dim + 1)]
        )
        return (
            *self.in_shapes[0][: self.start_dim],
            flattened_dim,
            *self.in_shapes[0][self.end_dim + 1 :],
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.flatten(x, start_dim=self.start_dim + 1, end_dim=self.end_dim + 1)


class TorchMixingWeightParameter(TorchUnaryParameterOp):
    def __init__(self, in_shape: tuple[int, ...], *, num_folds: int = 1):
        super().__init__(in_shape, num_folds=num_folds)
        if len(in_shape) != 2:
            raise ValueError(f"Expected shape (num_units, arity), but found {in_shape}")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.in_shape[0], self.in_shape[0] * self.in_shape[1]

    def forward(self, x: Tensor) -> Tensor:
        # x: (F, num_units, arity)
        # diag_weights: (arity, num_units, num_units)
        diag_weights = torch.vmap(torch.vmap(torch.diag, in_dims=1))(x)
        # (F, num_units, arity, num_units) -> (F, num_units, arity * num_units)
        return diag_weights.permute(0, 2, 1, 3).flatten(start_dim=2)


class TorchGaussianProductMean(TorchParameterOp):
    def __init__(
        self,
        in_mean1_shape: tuple[int, ...],
        in_stddev1_shape: tuple[int, ...],
        in_mean2_shape: tuple[int, ...],
        in_stddev2_shape: tuple[int, ...],
        *,
        num_folds: int = 1,
    ) -> None:
        assert in_mean1_shape == in_stddev1_shape
        assert in_mean2_shape == in_stddev2_shape
        super().__init__(
            in_mean1_shape, in_stddev1_shape, in_mean2_shape, in_stddev2_shape, num_folds=num_folds
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.in_shapes[0][0] * self.in_shapes[2][0],)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "in_mean1_shape": self.in_shapes[0],
            "in_stddev1_shape": self.in_shapes[1],
            "in_mean2_shape": self.in_shapes[2],
            "in_stddev2_shape": self.in_shapes[3],
        }

    def forward(self, mean1: Tensor, stddev1: Tensor, mean2: Tensor, stddev2: Tensor) -> Tensor:
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
        in_stddev1_shape: tuple[int, ...],
        in_stddev2_shape: tuple[int, ...],
        *,
        num_folds: int = 1,
    ) -> None:
        super().__init__(in_stddev1_shape, in_stddev2_shape, num_folds=num_folds)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.in_shapes[0][0] * self.in_shapes[1][0],)

    @property
    def config(self) -> dict[str, Any]:
        return {"in_stddev1_shape": self.in_shapes[0], "in_stddev2_shape": self.in_shapes[1]}

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
        in_mean1_shape: tuple[int, ...],
        in_stddev1_shape: tuple[int, ...],
        in_mean2_shape: tuple[int, ...],
        in_stddev2_shape: tuple[int, ...],
        *,
        num_folds: int = 1,
    ) -> None:
        assert in_mean1_shape == in_stddev1_shape
        assert in_mean2_shape == in_stddev2_shape
        super().__init__(
            in_mean1_shape, in_stddev1_shape, in_mean2_shape, in_stddev2_shape, num_folds=num_folds
        )
        self._log_two_pi = np.log(2.0 * np.pi)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.in_shapes[0][0] * self.in_shapes[2][0],)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "in_mean1_shape": self.in_shapes[0],
            "in_stddev1_shape": self.in_shapes[1],
            "in_mean2_shape": self.in_shapes[2],
            "in_stddev2_shape": self.in_shapes[3],
        }

    def forward(
        self,
        mean1: Tensor,
        stddev1: Tensor,
        mean2: Tensor,
        stddev2: Tensor,
    ) -> Tensor:
        var1 = torch.square(stddev1)  # (F, K1, C)
        var2 = torch.square(stddev2)  # (F, K2, C)
        var12 = var1.unsqueeze(dim=2) + var2.unsqueeze(dim=1)  # (F, K1, K2, C)
        inv_var12 = torch.reciprocal(var12)
        sq_mahalanobis = torch.square(mean1.unsqueeze(dim=2) - mean2.unsqueeze(dim=1)) * inv_var12
        log_partition = -0.5 * (self._log_two_pi + torch.log(var12) + sq_mahalanobis)
        return log_partition.view(-1, *self.shape)  # (F, K1 * K2, C)


class TorchPolynomialProduct(TorchBinaryParameterOp):
    # Use default __init__

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            self.in_shapes[0][0] * self.in_shapes[1][0],  # dim K
            self.in_shapes[0][1] + self.in_shapes[1][1] - 1,  # dim dp1
        )

    def forward(self, coeff1: Tensor, coeff2: Tensor) -> Tensor:
        # TODO: torch typing issue.
        fft: Callable[..., Tensor]  # type: ignore[misc]
        ifft: Callable[..., Tensor]  # type: ignore[misc]
        if coeff1.is_complex() or coeff2.is_complex():
            fft = torch.fft.fft
            ifft = torch.fft.ifft
        else:
            fft = torch.fft.rfft
            ifft = torch.fft.irfft

        degp1 = coeff1.shape[-1] + coeff2.shape[-1] - 1  # deg1p1 + deg2p1 - 1 = (deg1 + deg2) + 1.

        spec1 = fft(coeff1, n=degp1, dim=-1)  # shape (F, K1, dp1).
        spec2 = fft(coeff2, n=degp1, dim=-1)  # shape (F, K2, dp1).

        # shape (F, K1, 1, dp1), (F, 1, K2, dp1) -> (F, K1, K2, dp1) -> (F, K1*K2, dp1).
        spec = torch.flatten(
            spec1.unsqueeze(dim=2) * spec2.unsqueeze(dim=1), start_dim=1, end_dim=2
        )

        return ifft(spec, n=degp1, dim=-1)  # shape (F, K1*K2, dp1).


class TorchPolynomialDifferential(TorchUnaryParameterOp):
    def __init__(self, in_shape: tuple[int, ...], *, num_folds: int = 1, order: int = 1) -> None:
        if order <= 0:
            raise ValueError("The order of differentiation must be positive.")
        super().__init__(in_shape, num_folds=num_folds)
        self.order = order

    @property
    def shape(self) -> tuple[int, ...]:
        # if dp1>order, i.e., deg>=order, then diff, else const 0.
        return (
            self.in_shapes[0][0],
            self.in_shapes[0][1] - self.order if self.in_shapes[0][1] > self.order else 1,
        )

    @classmethod
    def _diff_once(cls, x: Tensor) -> Tensor:
        degp1 = x.shape[-1]  # x shape (F, K, dp1).
        arange = torch.arange(1, degp1).to(x)  # shape (deg,).
        return x[..., 1:] * arange  # a_n x^n -> n a_n x^(n-1), with a_0 disappeared.

    def forward(self, coeff: Tensor) -> Tensor:
        if coeff.shape[-1] <= self.order:
            return torch.zeros_like(coeff[..., :1])  # shape (F, K, 1).

        for _ in range(self.order):
            coeff = self._diff_once(coeff)
        return coeff  # shape (F, K, dp1-ord).
