from typing import Callable, List, Tuple, Union, cast
from typing_extensions import TypeVarTuple, Unpack  # FUTURE: in typing from 3.11

import torch
from torch import Tensor
from torch._functorch.eager_transforms import jvp  # TODO: jvp is not re-exported from torch.func.

Ts = TypeVarTuple("Ts")


# TODO: better typing? Ts is not bound to Tensor. if needed, there's no harm to change from last one
#       to the first one: Tuple[Tensor, Unpack[Ts]]
# NOTE: ellipsis is a class that can be recognized by type checkers but not runtime.
def batch_diff_at(
    func: Callable[[Tensor], Tuple[Unpack[Ts]]], indices: List[Union[int, slice, "ellipsis"]], /
) -> Callable[[Tensor], Tuple[Unpack[Ts], Tensor]]:
    """Transform a batched Tensor function into a function that calculates its differential w.r.t. \
    the element at the specified position in the input, i.e., a speficied column of its Jacobian.

    The dimension semantics of the input to func is specified by indices, which should slice all \
    of the batch dimension(s) but select only one element of other dimention(s). The func must be \
    batched on the same batch dimension(s) (allowed in different order), i.e., every output batch \
    depends solely on its own corresponding input batch.

    The differential will be in the same shape as the output of func, containing the partial of \
    output at each position w.r.t. the element at the specified position of the corresponding \
    input batch.

    For the convenience of higher-order differentials, here we accept a func that returns a tuple \
    of Tensor, and the transformed function will return all the original output along with the \
    differential of the last one.

    Args:
        func (Callable[[Tensor], Tuple[Unpack[Ts]]]): The batched Tensor function to be transformed.
        indices (List[Union[int, slice, ellipsis]]): The indices to select the element in the \
            input Tensor of func.

    Returns:
        Callable[[Tensor], Tuple[Unpack[Ts], Tensor]]: The transformed func that gives the output \
            of func and the required differential.
    """
    # We accept list as indices to mimic the subscript operator, but __getitem__ actually requires
    # a tuple like idx_slicing.
    idx_slicing = tuple(indices)

    def jvp_func(x: Tensor) -> Tuple[Unpack[Ts], Tensor]:
        """Wrap func to calculate the differential at given indices using Jac-vec prod.

        Args:
            x (Tensor): The input to func.

        Returns:
            Tuple[Unpack[Ts], Tensor]: The output of func, and the differential.
        """
        # This binary tangents sum up the Jacobian at elements selected by the 0/1 mask. Since func
        # is batched, it's guaranteed that only in-batch elements will be contribute non-zero values
        # in Jacobian. Thus, if indices correctly select only one element per-batch, jvp with this
        # tangents will leads to the correct differential at the given position.
        tangents = torch.zeros_like(x)
        tangents[idx_slicing] = 1
        # TODO: better annotation for jvp?
        # CAST: Return of jvp is not typed.
        # DISABLE: jvp can return tuple of 2 or 3, but here it's guaranteed to be 2.
        orig, (*_, diff) = cast(  # pylint: disable=unbalanced-tuple-unpacking
            Tuple[Tuple[Unpack[Ts]], Tuple[Tensor, ...]], jvp(func, (x,), (tangents,))
        )
        return orig + (diff,)

    return jvp_func


def batch_high_order_at(
    func: Callable[[Tensor], Tensor],
    x: Tensor,
    indices: List[Union[int, slice, "ellipsis"]],
    /,
    *,
    order: int,
) -> Tuple[Tensor, ...]:
    """Compute the high-order differentials of a batched Tensor function w.r.t. the element at the \
    specified position in the input.

    Args:
        func (Callable[[Tensor], Tensor]): The batched Tensor function.
        x (Tensor): The input to func.
        indices (List[Union[int, slice, ellipsis]]): The indices to select the element in x.
        order (int): The order of differentiation.

    Returns:
        Tuple[Tensor, ...]: The differentials from 0-order (original output) to the given order, \
            length order+1, each one's shape same as func output.
    """
    assert order > 0, "The order of differentiation must be positive."

    # ANNOTATE: The return value is allowed to be tuple of any length.
    diff_func: Callable[[Tensor], Tuple[Tensor, ...]] = lambda x: (func(x),)
    for _ in range(order):
        diff_func = batch_diff_at(diff_func, indices)

    return diff_func(x)
