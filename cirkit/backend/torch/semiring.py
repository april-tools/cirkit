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
from typing_extensions import TypeVarTuple, Unpack, final

import torch
from torch import Tensor

from cirkit.backend.torch.utils import flatten_dims, unflatten_dims

Ts = TypeVarTuple("Ts")
Semiring = TypeVar("Semiring", bound=Type["SemiringImpl"])


class SemiringImpl(ABC):
    """The abstract base class for semiring implementations.

    Due to numerical precision, the actual units in computational graph may hold values in, e.g., \
    log space, instead of linear space. And therefore, this provides a unified interface for the \
    computations so that computation can be done in a space suitable to the implementation \
    regardless of the global setting.
    """

    # A registry from semiring string identifiers to semiring class implementations
    _registry: ClassVar[Dict[str, Type["SemiringImpl"]]] = {}

    # A registry of morphisms between semiring class implementations
    _registry_morphisms: ClassVar[
        Dict[Tuple[Type["SemiringImpl"], Type["SemiringImpl"]], Callable[[Tensor], Tensor]]
    ] = {}

    @final
    @staticmethod
    def register(name: str) -> Callable[[Semiring], Semiring]:
        """Register a concrete semiring implementation by its name.

        Args:
            name (str): The name to register.

        Returns:
            Callable[[Semiring], Semiring]: The class decorator to register a subclass.
        """

        def _decorator(cls: Semiring) -> Semiring:
            """Register a concrete semiring implementation by its name.

            Args:
                cls (Semiring): The semiring subclass to register.

            Returns:
                Semiring: The class passed in.
            """
            # CAST: getattr gives Any.
            assert cast(
                bool, getattr(cls, "__final__", False)
            ), "Subclasses of SemiringImpl should be final."
            SemiringImpl._registry[name] = cls
            return cls

        return _decorator

    @final
    @classmethod
    def register_map_from(
        cls, other: Semiring
    ) -> Callable[[Callable[[Tensor], Tensor]], Callable[[Tensor], Tensor]]:
        """Register a concrete semiring morphism implementation.

        Args:
            other: The source semiring.

        Returns:
            Callable[[Callable[[Tensor], Tensor]], Callable[[Tensor], Tensor]]: The function decorator to register the morphism.
        """

        def _decorator(func: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
            """Register a concrete semiring morphism implementation.

            Args:
                func (Callable[[Tensor], Tensor]): The morphism between semirings to register.

            Returns:
                Callable[[Tensor], Tensor]: The morphism passed in.
            """
            SemiringImpl._registry_morphisms[(other, cls)] = func
            return func

        return _decorator

    @final
    @staticmethod
    def list() -> Iterable[str]:
        """List all semiring names registered.

        Returns:
            Iterable[str]: An iterable over all names available.
        """
        return iter(SemiringImpl._registry)

    @final
    @staticmethod
    def from_name(name: str) -> Semiring:
        """Get a semiring by its registered name.

        Args:
            name (str): The name to probe.

        Returns:
            Semiring: The retrieved concrete Semiring.
        """
        if name not in SemiringImpl._registry:
            raise IndexError(
                f"Unknown semiring '{name}'. Use @SemiringImpl.register(<name>) to register a new semiring"
            )
        return SemiringImpl._registry[name]

    @final
    @classmethod
    def map_from(cls, x: Tensor, semiring: Semiring) -> Tensor:
        """Map a tensor from the given semiring to `this` semiring.

        Args:
            x:
            semiring:

        Returns:

        """
        if cls == semiring:
            return x
        func: Optional[Callable[[Tensor], Tensor]] = SemiringImpl._registry_morphisms.get(
            (semiring, cls), None
        )
        if func is None:
            raise NotImplementedError(
                f"Semiring map from '{semiring.__name__}' to '{cls.__name__}' is not implemented"
            )
        return func(x)

    @final
    def __new__(cls) -> "SemiringImpl":
        """Raise an error when this class is instantiated.

        Raises:
            TypeError: When this class is instantiated.

        Returns:
            SemiringImpl: This method never returns.
        """
        raise TypeError("This class cannot be instantiated")

    @classmethod
    def einsum(
        cls,
        equation: str,
        *,
        inputs: Tuple[Tensor, ...],
        operands: Tuple[Tensor, ...],
        dim: Union[int, Sequence[int]],
        keepdim: bool,
    ) -> Tensor:
        operands = tuple(cls.cast(opd) for opd in operands)

        def _einsum_func(*xs: Tensor) -> Tensor:
            return torch.einsum(equation, *xs, *operands)

        return cls.apply(_einsum_func, *inputs, dim=dim, keepdim=keepdim)

    # NOTE: Subclasses should not touch any of the above final static methods but should implement
    #       all the following abstract class methods, and subclasses should be @final.

    @classmethod
    @abstractmethod
    def cast(cls, x: Tensor) -> Tensor:
        """Cast a tensor to the data type required by `this` semiring.

        Args:
            x: The tensor.

        Returns:
            Tensor: The tensor converted to the required data type.
        """

    @classmethod
    @abstractmethod
    def sum(
        cls, x: Tensor, /, *, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
    ) -> Tensor:
        """

        Args:
            x:
            dim:
            keepdim:

        Returns:

        """

    @classmethod
    @abstractmethod
    def add(cls, *xs: Tensor) -> Tensor:
        """

        Args:
            *xs:

        Returns:

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
        """Multiply broadcastable tensors.

        Args:
            *xs (Tensor): The input tensors, should have broadcastable shapes.

        Returns:
            Tensor: The multiply result.
        """

    # TODO: it's difficult to bound a variadic to Tensor with TypeVars, we can only use unbounded
    #       Unpack[TypeVarTuple].

    @classmethod
    @abstractmethod
    def apply(
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
        will not affect much. However, the input/output values may still be in another space, and \
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


@SemiringImpl.register("sum-product")
@final  # type: ignore[misc]
class SumProductSemiring(SemiringImpl):
    """The linear space computation."""

    @classmethod
    def cast(cls, x: Tensor) -> Tensor:
        """Cast a tensor to the data type required by the semiring.

        Args:
            x (Tensor): The tensor.

        Returns:
            Tensor: The tensor converted to the required data type.
        """
        if x.is_floating_point():
            return x
        raise ValueError(f"Cannot cast a tensor of type '{x.dtype}' to the '{cls.__name__}'")

    @classmethod
    def sum(
        cls, x: Tensor, /, *, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
    ) -> Tensor:
        dim = tuple(dim) if isinstance(dim, Sequence) else dim  # dim must be concrete type for sum.
        return x.sum(dim=dim, keepdim=keepdim)

    @classmethod
    def add(cls, *xs: Tensor) -> Tensor:
        raise functools.reduce(torch.add, xs)

    @classmethod
    def prod(
        cls, x: Tensor, /, *, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
    ) -> Tensor:
        # prod only accepts one dim and cannot be None.
        dims = dim if isinstance(dim, Sequence) else (dim,) if dim is not None else range(x.ndim)

        x = torch.prod(flatten_dims(x, dims=dims), dim=dims[0], keepdim=keepdim)
        if keepdim:  # We don't need to unflatten if not keepdim -- dims are just squeezed.
            # If we do keepdim, just "unqueeze" the 1s to the correct position.
            x = unflatten_dims(x, dims=dims, shape=(1,) * len(dims))

        return x

    @classmethod
    def mul(cls, *xs: Tensor) -> Tensor:
        return functools.reduce(torch.mul, xs)

    @classmethod
    def apply(
        cls,
        func: Callable[[Unpack[Ts]], Tensor],
        *xs: Unpack[Ts],
        dim: Union[int, Sequence[int]],
        keepdim: bool,
    ) -> Tensor:
        return func(*xs)


@SemiringImpl.register("lse-sum")
@final  # type: ignore[misc]
class LSESumSemiring(SemiringImpl):
    """The log space computation."""

    @classmethod
    def cast(cls, x: Tensor) -> Tensor:
        if x.is_floating_point():
            return x
        raise ValueError(f"Cannot cast a tensor of type '{x.dtype}' to the '{cls.__name__}'")

    @classmethod
    def sum(
        cls, x: Tensor, /, *, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
    ) -> Tensor:
        dim = tuple(dim) if isinstance(dim, Sequence) else dim  # dim must be concrete type for sum.
        return x.logsumexp(dim=dim, keepdim=keepdim)

    @classmethod
    def add(cls, *xs: Tensor) -> Tensor:
        return functools.reduce(torch.logaddexp, xs)

    @classmethod
    def prod(
        cls, x: Tensor, /, *, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
    ) -> Tensor:
        dim = tuple(dim) if isinstance(dim, Sequence) else dim  # dim must be concrete type for sum.
        return x.sum(dim=dim, keepdim=keepdim)

    @classmethod
    def mul(cls, *xs: Tensor) -> Tensor:
        return functools.reduce(torch.add, xs)

    @classmethod
    def apply(
        cls,
        func: Callable[[Unpack[Ts]], Tensor],
        *xs: Unpack[Ts],
        dim: Union[int, Sequence[int]],
        keepdim: bool,
    ) -> Tensor:
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


@SemiringImpl.register("complex-lse-sum")
@final  # type: ignore[misc]
class ComplexLSESumSemiring(SemiringImpl):
    """The complex log space computation."""

    @classmethod
    def cast(cls, x: Tensor) -> Tensor:
        if x.is_complex():
            return x
        if x.dtype == torch.float16:
            return x.to(torch.complex32)
        if x.dtype == torch.float32:
            return x.to(torch.complex64)
        if x.dtype == torch.float64:
            return x.to(torch.complex128)
        raise ValueError(f"Cannot cast a tensor of type '{x.dtype}' to the '{cls.__name__}'")

    @classmethod
    def sum(
        cls, x: Tensor, /, *, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
    ) -> Tensor:
        dim = tuple(dim) if isinstance(dim, Sequence) else dim  # dim must be concrete type for sum.
        return x.logsumexp(dim=dim, keepdim=keepdim)

    @classmethod
    def add(cls, *xs: Tensor) -> Tensor:
        return functools.reduce(torch.logaddexp, xs)

    @classmethod
    def prod(
        cls, x: Tensor, /, *, dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
    ) -> Tensor:
        dim = tuple(dim) if isinstance(dim, Sequence) else dim  # dim must be concrete type for sum.
        return x.sum(dim=dim, keepdim=keepdim)

    @classmethod
    def mul(cls, *xs: Tensor) -> Tensor:
        return functools.reduce(torch.add, xs)

    @classmethod
    def apply(
        cls,
        func: Callable[[Unpack[Ts]], Tensor],
        *xs: Unpack[Ts],
        dim: Union[int, Sequence[int]],
        keepdim: bool,
    ) -> Tensor:
        dims = tuple(dim) if isinstance(dim, Sequence) else (dim,)

        # NOTE: Due to usage of intermediate results, they need to be instantiated in lists but not
        #       generators, because generators can't save much if we want to reuse.
        # CAST: Expected tuple of Tensor but got Ts.
        x = [cast(Tensor, xi) for xi in xs]
        # We need flatten because max only works on one dim, and then match shape for exp_x.
        max_x = [
            unflatten_dims(
                torch.max(flatten_dims(xi.real, dims=dims), dim=dims[0], keepdim=True)[0],
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
        return ComplexLSESumSemiring._grad_safe_complex_log(func_exp_x) + sum_max_x

    @classmethod
    def _grad_safe_complex_log(cls, x: Tensor) -> Tensor:
        # Compute log(x) safely where x is a complex tensor.
        ComplexLSESumSemiring.__double_zero_clamp_(x.real)
        return torch.log(x)

    @staticmethod
    @torch.no_grad()
    @torch.compile()
    def __double_zero_clamp_(x: Tensor) -> None:
        eps = torch.finfo(torch.get_default_dtype()).tiny
        close_zero_mask = (x > -eps) & (x < eps)
        clamped_x = eps * (1.0 - 2.0 * torch.signbit(x))
        torch.where(close_zero_mask, clamped_x, x, out=x)


@SumProductSemiring.register_map_from(LSESumSemiring)
def _(x: Tensor) -> Tensor:
    return torch.exp(x)


@SumProductSemiring.register_map_from(ComplexLSESumSemiring)
def _(x: Tensor) -> Tensor:
    if torch.all(torch.isreal(x)):
        return torch.exp(x.real)
    raise ValueError(
        f"Cannot map a tensor with non-zero imaginary part to {SumProductSemiring.__name__}"
    )


@LSESumSemiring.register_map_from(SumProductSemiring)
def _(x: Tensor) -> Tensor:
    return torch.log(x)


@LSESumSemiring.register_map_from(ComplexLSESumSemiring)
def _(x: Tensor) -> Tensor:
    if torch.all(torch.isreal(x)):
        return x.real
    raise ValueError(
        f"Cannot map a tensor with non-zero imaginary part to {LSESumSemiring.__name__}"
    )


@ComplexLSESumSemiring.register_map_from(SumProductSemiring)
def _(x: Tensor) -> Tensor:
    return torch.log(ComplexLSESumSemiring.cast(x))


@ComplexLSESumSemiring.register_map_from(LSESumSemiring)
def _(x: Tensor) -> Tensor:
    return ComplexLSESumSemiring.cast(x)
