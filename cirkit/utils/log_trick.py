from typing import Callable, List, overload

import torch
from torch import Tensor


@overload
def log_func_exp(
    x: Tensor, /, *, func: Callable[[Tensor], Tensor], dim: int, keepdim: bool
) -> Tensor:
    ...


@overload
def log_func_exp(
    x0: Tensor, x1: Tensor, /, *, func: Callable[[Tensor, Tensor], Tensor], dim: int, keepdim: bool
) -> Tensor:
    ...


# TODO: is there a way to avoid the overloads and type func? or do we only accept these two?
def log_func_exp(  # type: ignore[misc]
    *x: Tensor, func: Callable[..., Tensor], dim: int, keepdim: bool
) -> Tensor:
    """Perform the log-sum-exp trick extended to any given function.

    The provided function should be linear and behave consistently with `dim` and `keepdim` -- \
    there's no check on these conditions and users should guarantee it's valid.

    Args:
        x (Tensor): The input to `func`, can be multiple `Tensor`s.
        func (Callable[..., Tensor]): The function generalizing `torch.sum`.
        dim (int): The dimension that is collapsed by the `sum`-like operation.
        keepdim (bool): Whether to keep `dim` as a size-1 dim.

    Returns:
        Tensor: The result of `log(func(exp(x)))`.
    """
    # TODO: max type should be fixed by the next pytorch
    max_x: List[Tensor] = [
        torch.max(xi, dim=dim, keepdim=True)[0] for xi in x  # type: ignore[misc]
    ]
    exp_x = [torch.exp(xi - xi_max) for xi, xi_max in zip(x, max_x)]

    func_exp_x = func(*exp_x)

    # TODO: verify the behavior of `sum` under torch.compile. this part may need rewrite
    sum_max_x = sum(max_x[1:], max_x[0])  # write in this way to avoid redundant +0
    if not keepdim:
        sum_max_x = sum_max_x.squeeze(dim)

    log_func_exp_x = torch.log(func_exp_x) + sum_max_x

    return log_func_exp_x
