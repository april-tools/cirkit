import itertools
from typing import Callable

import pytest
import torch
from torch import Tensor

from cirkit.new.utils import batch_high_order_at
from tests import floats

B, N, M = 5, 10, 2  # The shape of x.


def batched_func(x: Tensor) -> Tensor:
    return torch.exp(x.mT @ x)


def _bw_1st(func: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Implement 1st order diff using simple backward.

    Args:
        func (Callable[[Tensor], Tensor]): The function, shape (*in,) -> (*out,).
        x (Tensor): The input, shape (*in,).

    Returns:
        Tensor: The total diff, shape (*out, *in).
    """
    output = func(x)
    out_shape = output.shape
    output = output.view(-1)

    diff = x.new_zeros(output.nelement(), *x.shape)

    for i in range(output.nelement()):
        diff[i] = torch.autograd.grad(output[i], x, create_graph=True)[0]

    return diff.view(out_shape + x.shape)


def _bw_2nd(func: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Implement 2nd order diff (of same var) using simple backward.

    Args:
        func (Callable[[Tensor], Tensor]): The function, shape (*in,) -> (*out,).
        x (Tensor): The input, shape (*in,).

    Returns:
        Tensor: The total diff, shape (*out, *in).
    """
    output = func(x)
    out_shape = output.shape
    output = output.view(-1)

    diff = x.new_zeros(output.nelement(), x.nelement())

    for i in range(output.nelement()):
        diff_1st = torch.autograd.grad(output[i], x, create_graph=True)[0]
        for j in range(x.nelement()):
            # NOTE: This grad output is not contiguous, so the second view must be reshape.
            diff[i, j] = torch.autograd.grad(diff_1st.view(-1)[j], x, create_graph=True)[0].reshape(
                -1
            )[j]

    return diff.view(out_shape + x.shape)


# NOTE: We only test batch_high_order_at() here and it should be enough to guarantee batch_diff_at.


@pytest.mark.parametrize("order", [1, 2, 3])
def test_batch_diff_orig(order: int) -> None:
    x = torch.rand(B, N, M, requires_grad=True)
    func_x = batched_func(x)  # shape (B, M, M).

    for i, j in itertools.product(range(N), range(M)):
        diffs = batch_high_order_at(batched_func, x, [slice(None), i, j], order=order)
        assert floats.allclose(diffs[0], func_x)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_batch_diff_1st(order: int) -> None:
    x = torch.rand(B, N, M, requires_grad=True)
    # shape (B, M, M, B, N, M) -> (M, M, N, M, B) -> (B, M, M, N, M).
    diff_1st = _bw_1st(batched_func, x).diagonal(dim1=0, dim2=3).movedim(-1, 0)

    for i, j in itertools.product(range(N), range(M)):
        diffs = batch_high_order_at(batched_func, x, [slice(None), i, j], order=order)
        assert floats.allclose(diffs[1], diff_1st[..., i, j])


@pytest.mark.parametrize("order", [2, 3])
def test_batch_diff_2nd(order: int) -> None:
    x = torch.rand(B, N, M, requires_grad=True)
    # shape (B, M, M, B, N, M) -> (M, M, N, M, B) -> (B, M, M, N, M).
    diff_2nd = _bw_2nd(batched_func, x).diagonal(dim1=0, dim2=3).movedim(-1, 0)

    for i, j in itertools.product(range(N), range(M)):
        diffs = batch_high_order_at(batched_func, x, [slice(None), i, j], order=order)
        assert floats.allclose(diffs[2], diff_2nd[..., i, j])
