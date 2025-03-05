import itertools
from collections.abc import Sequence
from typing import Any, Callable, Mapping, Protocol, cast

import torch
from torch import Tensor, autograd, nn


class SafeLog(autograd.Function):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return torch.log(x)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        (x,) = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return torch.nan_to_num(grad_output / x)


safelog = SafeLog.apply


class ComplexSafeLog(autograd.Function):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return torch.log(x)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        (x,) = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        return torch.nan_to_num(grad_output / x.conj())


csafelog = ComplexSafeLog.apply


def flatten_dims(x: Tensor, /, *, dims: Sequence[int]) -> Tensor:
    """Flatten the given dims in the input.

    If the dims are not continuous, they will be permuted and flattened to the position of the \
    first element in dims.

    Intended to be used as a helper for some torch functions that can only work on one dim.

    Args:
        x (Tensor): The tensor to be flattened.
        dims (Sequence[int]): The dimensions to flatten along, expected to be sorted.

    Returns:
        Tensor: The flattened tensor.
    """
    if not dims:  # When dims[0] does not work.
        return x

    start_dim, end_dim = dims[0], dims[0] + len(dims)
    # Note that for flatten, end_dim is inclusive.
    return x.movedim(tuple(dims), tuple(range(start_dim, end_dim))).flatten(start_dim, end_dim - 1)


def unflatten_dims(x: Tensor, /, *, dims: Sequence[int], shape: Sequence[int]) -> Tensor:
    """Unflatten the first dim in dims in the input to get a given shape.

    This is the inverse transformation of flatten_dims, provided a correspondimg shape.

    Args:
        x (Tensor): The tensor to be unflattened.
        dims (Sequence[int]): The dimensions to unflatten to, should be the same as flatten_dims.
        shape (Sequence[int]): The shape to unflatten to, can be either the shape for dims, or the \
            whole shape for the output. If the latter, the shape will not be checked for \
            consistency outside dims.

    Returns:
        Tensor: The unflattened tensor.
    """
    if not dims:  # When dims[0] does not work.
        return x

    # We require dims to be sorted so that there's no ambiguation in how shape is interpreted,
    # unless the shape itself never causes ambiguation.
    assert all(s == 1 for s in shape) or all(
        l < r for l, r in itertools.pairwise(dims)
    ), "dims must be sorted for unflatten_dims."

    if len(shape) == x.ndim - 1 + len(dims):  # The shape is for whole output.
        shape = [shape[d] for d in dims]
    # The shape is now for dims.

    start_dim, end_dim = dims[0], dims[0] + len(dims)
    # TODO: x.unflatten is not typed, must use torch.unflatten for now.
    return torch.unflatten(x, dim=start_dim, sizes=shape).movedim(
        tuple(range(start_dim, end_dim)), tuple(dims)
    )


class GateFunction(Protocol):
    def __call__(
        self, shape: tuple[int, ...], *args: list[Any], **kwargs: Mapping[str, Any]
    ) -> Tensor:
        """
        The interface that a gate function must implement.
        It takes a shape as input, which is the shape of the expected
        output from the function.

        Args:
            shape (tuple[int, ...]): The shape expected as input.
            *args (list[Any]): The positional arguments for the function.
            **kwargs (Mapping[str, Any]): The keyword arguments for the function.

        Returns:
            Tensor: The tensor of shape `shape` that will be used as parameter.
        """
        ...


class CachedGateFunctionEval(object):
    def __init__(self, function_id: str, gate_function: GateFunction):
        super().__init__()
        self._function_id = function_id
        self._gate_function = gate_function
        self._cached_output: dict[str, Tensor] | None = None

    @property
    def function_id(self) -> str:
        return self._function_id

    @property
    def gate_function(self) -> GateFunction:
        return self._gate_function

    def reset_cache(self):
        self._cached_output = None

    def memoize(self, shape: tuple[int, ...], *args, **kwargs):
        self._cached_output = self.gate_function(shape, *args, **kwargs)

    def __call__(self) -> Tensor:
        if self._cached_output is None:
            raise ValueError("No output mapping is stored. Call memoize() method first")
        return self._cached_output
