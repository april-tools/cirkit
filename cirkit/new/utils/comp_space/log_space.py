import functools
from typing import Callable, Optional, Sequence, Tuple, Union, cast
from typing_extensions import final  # FUTURE: in typing from 3.11 for __final__
from typing_extensions import TypeVarTuple, Unpack  # FUTURE: in typing from 3.11

import torch
from torch import Tensor

from cirkit.new.utils.comp_space.comp_space import ComputationSapce
from cirkit.new.utils.flatten import flatten_dims, unflatten_dims

Ts = TypeVarTuple("Ts")


# TODO: ignore is bug?
# IGNORE: The annotation for final in typeshed/typing_extensions.pyi contains Any.
@ComputationSapce.register("log")
@final  # type: ignore[misc]
class LogSpace(ComputationSapce):
    """The log space computation."""

    @classmethod
    def from_log(cls, x: Tensor) -> Tensor:
        """Convert a value from log space to the current space.

        Args:
            x (Tensor): The value in log space.

        Returns:
            Tensor: The value in the current space.
        """
        return x

    @classmethod
    def from_linear(cls, x: Tensor) -> Tensor:
        """Convert a value from linear space to the current space.

        Args:
            x (Tensor): The value in linear space.

        Returns:
            Tensor: The value in the current space.
        """
        return torch.log(x)

    @classmethod
    def sum(
        cls,
        func: Callable[[Unpack[Ts]], Tensor],
        *xs: Unpack[Ts],
        dim: Union[int, Sequence[int]],
        keepdim: bool,
    ) -> Tensor:
        """Apply a sum-like functions to the tensor(s).

        In log space, we apply the log-func-exp trick, an extension to log-sum-exp.

        The sum units may perform not just plain sum, but also weighted sum or even einsum. In \
        fact, it can possibly be any function that is linear to the each input. All that kind of \
        func can be used here.

        It is expected that func always does computation in the linear space, as with numerical \
        tricks, only relatively significant numbers contribute the final answer, and underflow \
        will not affect much. However the input/output values may still be in another space, and \
        needs to be projected here.

        Args:
            func (Callable[[Unpack[Ts]], Tensor]): The sum-like function to be applied.
            *xs (Unpack[Ts]): The input tensors. Type expected to be Tensor.
            dim (Union[int, Sequence[int]]): The dimension(s) along which the values are \
                correlated and must be scaled together, i.e., the dim(s) to sum along. This should \
                match the actual operation done by func. The same dim is shared among all inputs.
            keepdim (bool): Whether the dim is kept as a size-1 dim, should match the actual \
                operation done by func.

        Returns:
            Tensor: The sum result.
        """
        dims = tuple(dim) if isinstance(dim, Sequence) else (dim,)

        # NOTE: Due to usage of intermediate results, they need to be instantiated in lists but not
        #       generators, because generators can't save much if we want to reuse.
        # CAST: Expected tuple of Tensor but got Ts.
        x = [cast(Tensor, xi) for xi in xs]
        # We need flatten because max only works on one dim, and then match shape for exp_x.
        max_x = [
            unflatten_dims(
                torch.max(flatten_dims(xi, dims=dims), dim=dims[0], keepdim=True)[0],
                dims=dims,
                shape=(1,) * len(dims),  # The size for dims is 1 after max.
            )
            for xi in x
        ]
        exp_x = [torch.exp(xi - xi_max) for xi, xi_max in zip(x, max_x)]

        # NOTE: exp_x is not tuple, but list still can be unpacked with *.
        # CAST: Expected Ts but got tuple (actually list) of Tensor.
        func_exp_x = func(*cast(Tuple[Unpack[Ts]], exp_x))

        # TODO: verify the behavior of reduce under torch.compile
        sum_max_x = functools.reduce(torch.add, max_x)  # Do n-1 add instead of n.
        if not keepdim:
            sum_max_x = sum_max_x.squeeze(dims)  # To match shape of func_exp_x.
        log_func_exp_x = torch.log(func_exp_x) + sum_max_x

        return log_func_exp_x

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
        dim = tuple(dim) if isinstance(dim, Sequence) else dim  # dim must be concrete type for sum.
        return x.sum(dim=dim, keepdim=keepdim)

    @classmethod
    def mul(cls, *xs: Tensor) -> Tensor:
        """Do the multiply among broadcastable tensors.

        Args:
            *xs (Tensor): The input tensors, should have broadcastable shapes.

        Returns:
            Tensor: The multiply result.
        """
        return functools.reduce(torch.add, xs)
