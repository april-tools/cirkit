import functools
from typing import Callable, Optional, Sequence, Union
from typing_extensions import final  # FUTURE: in typing from 3.11 for __final__
from typing_extensions import TypeVarTuple, Unpack  # FUTURE: in typing from 3.11

import torch
from torch import Tensor

from cirkit.new.utils.comp_space.comp_space import ComputationSapce
from cirkit.new.utils.flatten import flatten_dims, unflatten_dims

Ts = TypeVarTuple("Ts")


# TODO: ignore is bug?
# IGNORE: The annotation for final in typeshed/typing_extensions.pyi contains Any.
@ComputationSapce.register("linear")
@final  # type: ignore[misc]
class LinearSpace(ComputationSapce):
    """The linear space computation."""

    @classmethod
    def from_log(cls, x: Tensor) -> Tensor:
        """Convert a value from log space to the current space.

        Args:
            x (Tensor): The value in log space.

        Returns:
            Tensor: The value in the current space.
        """
        return torch.exp(x)

    @classmethod
    def from_linear(cls, x: Tensor) -> Tensor:
        """Convert a value from linear space to the current space.

        Args:
            x (Tensor): The value in linear space.

        Returns:
            Tensor: The value in the current space.
        """
        return x

    @classmethod
    def sum(
        cls,
        func: Callable[[Unpack[Ts]], Tensor],
        *xs: Unpack[Ts],
        dim: Union[int, Sequence[int]],
        keepdim: bool,
    ) -> Tensor:
        """Apply a sum-like functions to the tensor(s).

        In linear space, we simply apply func to the inputs, and the dim is ignored.

        Args:
            func (Callable[[Unpack[Ts]], Tensor]): The sum-like function to be applied.
            *xs (Unpack[Ts]): The input tensors. Type expected to be Tensor.
            dim (Union[int, Sequence[int]]): Ignored.
            keepdim (bool): Ignored.

        Returns:
            Tensor: The sum result.
        """
        return func(*xs)

    @classmethod
    def prod(
        cls, x: Tensor, /, *, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
    ) -> Tensor:
        """Do the product within a tensor on given dim(s).

        Args:
            x (Tensor): The input tensor.
            dim (Optional[Union[int, Sequence[int]]], optional): The dimension(s) to reduce along, \
                None for all dims. Defaults to None.
            keepdim (bool, optional): Whether the dim is kept as a size-1 dim. Defaults to False.

        Returns:
            Tensor: The product result.
        """
        # prod only accepts one dim and cannot be None.
        dims = dim if isinstance(dim, Sequence) else (dim,) if dim is not None else range(x.ndim)

        x = torch.prod(flatten_dims(x, dims=dims), dim=dims[0], keepdim=keepdim)
        if keepdim:  # We don't need to unflatten if not keepdim -- dims are just squeezed.
            # If we do keepdim, just "unqueeze" the 1s to the correct position.
            x = unflatten_dims(x, dims=dims, shape=(1,) * len(dims))

        return x

    @classmethod
    def mul(cls, *xs: Tensor) -> Tensor:
        """Do the multiply among broadcastable tensors.

        Args:
            *xs (Tensor): The input tensors, should have broadcastable shapes.

        Returns:
            Tensor: The multiply result.
        """
        return functools.reduce(torch.mul, xs)
