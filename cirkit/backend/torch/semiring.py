import functools
from abc import ABC, abstractmethod
from typing import (
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from typing_extensions import Self, TypeVarTuple, Unpack, final

import torch
from torch import Tensor

from cirkit.backend.torch.utils import flatten_dims, unflatten_dims

Ts = TypeVarTuple("Ts")
SemiringCls = TypeVar("SemiringCls", bound=Type["Semiring"])


class Semiring(ABC):
    """The abstract base class for compotational spaces.

    Due to numerical precision, the actual units in computational graph may hold values in, e.g., \
    log space, instead of linear space. And therefore, this provides a unified interface for the \
    computations so that computation can be done in a space suitable to the implementation \
    regardless of the global setting.
    """

    _registry: ClassVar[Dict[str, Type["Semiring"]]] = {}

    @final
    @staticmethod
    def register(name: str) -> Callable[[SemiringCls], SemiringCls]:
        """Register a concrete Semiring implementation by its name.

        Args:
            name (str): The name to register.

        Returns:
            Callable[[CompSpaceClsT], CompSpaceClsT]: The class decorator to register a subclass.
        """

        def _decorator(cls: SemiringCls) -> SemiringCls:
            """Register a concrete Semiring implementation by its name.

            Args:
                cls (CompSpaceClsT): The Semiring subclass to register.

            Returns:
                CompSpaceClsT: The class passed in.
            """
            # CAST: getattr gives Any.
            assert cast(
                bool, getattr(cls, "__final__", False)
            ), "Subclasses of Semiring should be final."
            Semiring._registry[name] = cls
            return cls

        return _decorator

    @final
    @staticmethod
    def list_all_comp_space() -> Iterable[str]:
        """List all Semiring names registered.

        Returns:
            Iterable[str]: An iterable over all names available.
        """
        return iter(Semiring._registry)

    @final
    @staticmethod
    def from_name(name: str) -> Type["Semiring"]:
        """Get a Semiring by its registered name.

        Args:
            name (str): The name to probe.

        Returns:
            Type[Semiring]: The retrieved concrete Semiring.
        """
        if name not in Semiring._registry:
            raise IndexError(
                f"Unknown semiring '{name}'. Use @Semiring.register(<name>) to register a new semiring"
            )
        return Semiring._registry[name]

    # TODO: Never should be used. This is known issue: https://github.com/python/mypy/issues/14044
    @final
    def __new__(cls) -> Self:
        """Raise an error when this class is instantiated.

        Raises:
            TypeError: When this class is instantiated.

        Returns:
            Self: This method never returns.
        """
        raise TypeError("This class cannot be instantiated.")

    # NOTE: Subclasses should not touch any of the above final static methods but should implement
    #       all of the following abstract class methods, and subclasses should be @final.

    # TODO: if needed, we can also have to_log, to_lin.
    @classmethod
    @abstractmethod
    def from_lse_sum(cls, x: Tensor) -> Tensor:
        """Convert a value from log space to the current space.

        Args:
            x (Tensor): The value in log space.

        Returns:
            Tensor: The value in the current space.
        """

    @classmethod
    @abstractmethod
    def from_sum_product(cls, x: Tensor) -> Tensor:
        """Convert a value from linear space to the current space.

        Args:
            x (Tensor): The value in linear space.

        Returns:
            Tensor: The value in the current space.
        """

    # TODO: it's difficult to bound a variadic to Tensor with TypeVars, we can only use unbounded
    #       Unpack[TypeVarTuple].
    @classmethod
    @abstractmethod
    def sum(
        cls,
        func: Callable[[Unpack[Ts]], Tensor],
        *xs: Unpack[Ts],
        dim: Union[int, Sequence[int]],
        keepdim: bool,
    ) -> Tensor:
        """Apply a sum-like functions to the tensor(s).

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

    @classmethod
    @abstractmethod
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

    @classmethod
    @abstractmethod
    def mul(cls, *xs: Tensor) -> Tensor:
        """Do the multiply among broadcastable tensors.

        Args:
            *xs (Tensor): The input tensors, should have broadcastable shapes.

        Returns:
            Tensor: The multiply result.
        """


@Semiring.register("sum-product")
@final  # type: ignore[misc]
class SumProductSemiring(Semiring):
    """The linear space computation."""

    @classmethod
    def from_lse_sum(cls, x: Tensor) -> Tensor:
        """Convert a value from log space to the current space.

        Args:
            x (Tensor): The value in log space.

        Returns:
            Tensor: The value in the current space.
        """
        return torch.exp(x)

    @classmethod
    def from_sum_product(cls, x: Tensor) -> Tensor:
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


@Semiring.register("lse-sum")
@final  # type: ignore[misc]
class LSESumSemiring(Semiring):
    """The log space computation."""

    @classmethod
    def from_lse_sum(cls, x: Tensor) -> Tensor:
        """Convert a value from log space to the current space.

        Args:
            x (Tensor): The value in log space.

        Returns:
            Tensor: The value in the current space.
        """
        return x

    @classmethod
    def from_sum_product(cls, x: Tensor) -> Tensor:
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
