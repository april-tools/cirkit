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
        dim: int,
        keepdim: bool,
    ) -> Tensor:
        operands = tuple(cls.cast(opd) for opd in operands)

        def _einsum_func(*xs: Tensor) -> Tensor:
            return torch.einsum(equation, *xs, *operands)

        return cls.apply_reduce(_einsum_func, *inputs, dim=dim, keepdim=keepdim)

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
    def apply_reduce(
        cls,
        func: Callable[[Unpack[Ts]], Tensor],
        *xs: Unpack[Ts],
        dim: int,
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
            dim (int): The dimension along which the values are \
                correlated and must be scaled together, i.e., the dim to sum along. This should \
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
    def sum(cls, x: Tensor, /, *, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        return x.sum(dim=dim, keepdim=keepdim)

    @classmethod
    def add(cls, *xs: Tensor) -> Tensor:
        raise functools.reduce(torch.add, xs)

    @classmethod
    def prod(cls, x: Tensor, /, *, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        # prod only accepts one dim and cannot be None.
        dim = dim if dim is not None else range(x.ndim)
        return torch.prod(x, dim=dim, keepdim=keepdim)

    @classmethod
    def mul(cls, *xs: Tensor) -> Tensor:
        return functools.reduce(torch.mul, xs)

    @classmethod
    def apply_reduce(
        cls,
        func: Callable[[Unpack[Ts]], Tensor],
        *xs: Unpack[Ts],
        dim: int,
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
    def sum(cls, x: Tensor, /, *, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        return x.logsumexp(dim=dim, keepdim=keepdim)

    @classmethod
    def add(cls, *xs: Tensor) -> Tensor:
        return functools.reduce(torch.logaddexp, xs)

    @classmethod
    def prod(cls, x: Tensor, /, *, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        dim = tuple(dim) if isinstance(dim, Sequence) else dim  # dim must be concrete type for sum.
        return x.sum(dim=dim, keepdim=keepdim)

    @classmethod
    def mul(cls, *xs: Tensor) -> Tensor:
        return functools.reduce(torch.add, xs)

    @classmethod
    def apply_reduce(
        cls,
        func: Callable[[Unpack[Ts]], Tensor],
        *xs: Unpack[Ts],
        dim: int,
        keepdim: bool,
    ) -> Tensor:
        # NOTE: Due to usage of intermediate results, they need to be instantiated in lists but not
        #       generators, because generators can't save much if we want to reuse.
        # CAST: Expected tuple of Tensor but got Ts.
        xs = [cast(Tensor, xi) for xi in xs]
        max_xs = [torch.max(xi, dim=dim, keepdim=True)[0] for xi in xs]
        exp_xs = [torch.exp(xi - max_xi) for xi, max_xi in zip(xs, max_xs)]

        # NOTE: exp_x is not tuple, but list still can be unpacked with *.
        # CAST: Expected Ts but got tuple (actually list) of Tensor.
        func_exp_xs = func(*cast(Tuple[Unpack[Ts]], exp_xs))

        # TODO: verify the behavior of reduce under torch.compile
        reduced_max_xs = functools.reduce(torch.add, max_xs)  # Do n-1 add instead of n.
        if not keepdim:
            reduced_max_xs = reduced_max_xs.squeeze(dim)  # To match shape of func_exp_x.
        return torch.log(func_exp_xs) + reduced_max_xs


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
    def sum(cls, x: Tensor, /, *, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        return x.logsumexp(dim=dim, keepdim=keepdim)

    @classmethod
    def add(cls, *xs: Tensor) -> Tensor:
        return functools.reduce(torch.logaddexp, xs)

    @classmethod
    def prod(cls, x: Tensor, /, *, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        return x.sum(dim=dim, keepdim=keepdim)

    @classmethod
    def mul(cls, *xs: Tensor) -> Tensor:
        return functools.reduce(torch.add, xs)

    @classmethod
    def apply_reduce(
        cls,
        func: Callable[[Unpack[Ts]], Tensor],
        *xs: Unpack[Ts],
        dim: int,
        keepdim: bool,
    ) -> Tensor:
        # NOTE: Due to usage of intermediate results, they need to be instantiated in lists but not
        #       generators, because generators can't save much if we want to reuse.
        # CAST: Expected tuple of Tensor but got Ts.
        xs = [cast(Tensor, xi) for xi in xs]
        max_xs = [torch.max(xi.real, dim=dim, keepdim=True)[0] for xi in xs]
        exp_xs = [torch.exp(xi - max_xi) for xi, max_xi in zip(xs, max_xs)]

        # NOTE: exp_x is not tuple, but list still can be unpacked with *.
        # CAST: Expected Ts but got tuple (actually list) of Tensor.
        func_exp_xs = func(*cast(Tuple[Unpack[Ts]], exp_xs))

        # TODO: verify the behavior of reduce under torch.compile
        reduced_max_xs = functools.reduce(torch.add, max_xs)  # Do n-1 add instead of n.
        if not keepdim:
            reduced_max_xs = reduced_max_xs.squeeze(dim)  # To match shape of func_exp_x.

        # Compute log(x) safely where x is a complex tensor.
        # The problem is that if x = 0 + 0j, then the complex gradient of log(x) yields NaNs.
        # Note that for real non-monotonic circuits this problem cannot be avoided by simply
        # clipping the parameters of e.g., dense layers. In fact, even if we clipped the parameters
        # to be sufficiently far from zero, cancellations would still arise from negations, which
        # in turn might result in underflows. This has been observed in float32 for squared
        # non-monotonic PCs with real parameters. To solve this issue, here we clamp the real
        # part to be at least the tiny value of the default float dtype precision, in absolute.
        # Note that we do not need to clamp also the imaginary parts, since the complex gradients of
        # log(v + 0j) are 'well-behaved' for every non-zero real value 'v'.
        # Furthermore, torch.compile is used to hopefully obtain a faster kernel.
        # NOTE: to reproduce the bug, place the following assertion in the above code.
        #       This checks whether the real output of the function in the 'apply_reduce' is not 0.0.
        # assert not torch.any(torch.isclose(func_exp_xs.real.sign(), torch.tensor(0.0, device=func_exp_xs.device)))
        ComplexLSESumSemiring._double_zero_clamp_(func_exp_xs.real)
        return torch.log(func_exp_xs) + reduced_max_xs

    @staticmethod
    @torch.no_grad()
    @torch.compile()
    def _double_zero_clamp_(x: Tensor) -> None:
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
