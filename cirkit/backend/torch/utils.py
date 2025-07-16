import itertools
from collections.abc import Sequence
from typing import Any, Callable

import torch
from torch import Tensor, autograd


# pylint: disable-next=abstract-method
class SafeLog(autograd.Function):
    @staticmethod
    def forward(x: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        return torch.log(x)

    @staticmethod
    def setup_context(  # pylint: disable=arguments-differ
        ctx: Any, inputs: tuple[Tensor, ...], output: Tensor
    ) -> None:
        (x,) = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        (x,) = ctx.saved_tensors
        return torch.nan_to_num(grad_output / x)


safelog = SafeLog.apply


# pylint: disable-next=abstract-method
class ComplexSafeLog(autograd.Function):
    @staticmethod
    def forward(x: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        return torch.log(x)

    @staticmethod
    def setup_context(  # pylint: disable=arguments-differ
        ctx: Any, inputs: tuple[Tensor, ...], output: Tensor
    ) -> None:
        (x,) = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        (x,) = ctx.saved_tensors
        return torch.nan_to_num(grad_output / x.conj())


csafelog: Callable[[Tensor], Tensor] = ComplexSafeLog.apply


def flatten_dims(x: Tensor, /, *, dims: Sequence[int]) -> Tensor:
    """Flatten the given dims in the input.

    If the dims are not continuous, they will be permuted and flattened to the position of the \
    first element in dims.

    Intended to be used as a helper for some torch functions that can only work on one dim.

    Args:
        x: The tensor to be flattened.
        dims: The dimensions to flatten along, expected to be sorted.

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
