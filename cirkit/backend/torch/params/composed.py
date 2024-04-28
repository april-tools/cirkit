import functools
from abc import abstractmethod
from functools import cached_property
from typing import Callable, Generic, List, Optional, Protocol, Tuple, cast
from typing_extensions import TypeVarTuple, Unpack  # FUTURE: in typing from 3.11

import torch
from torch import Tensor, nn

from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.utils import flatten_dims, unflatten_dims


class _TensorFuncWithDim(Protocol):
    """The protocol for `(Tensor, dim: int) -> Tensor`."""

    def __call__(self, x: Tensor, /, dim: int) -> Tensor:
        ...


class _HasDimsTuple(Protocol):
    """The protocol providing self.dims.

    See: https://mypy.readthedocs.io/en/stable/more_types.html?highlight=reveal_type#mixin-classes.
    """

    # DISABLE: It's intended to omit method docstring for this Protocol.
    @property
    @abstractmethod
    # pylint: disable-next=missing-function-docstring
    def dims(self) -> Tuple[int, ...]:
        ...


class _ApplySingleDimFuncMixin:
    """A mixin with a helper method that applies a function accepting a single dim to multiple \
    dims specified by self.dims.

    This is useful for NormalizedReparam with certain normalizers.
    """

    def _apply_single_dim_func(
        self: _HasDimsTuple, func: _TensorFuncWithDim, x: Tensor, /
    ) -> Tensor:
        """Apply a tensor function accepting a single dim over self.dims.

        Args:
            func (_TensorFuncWithDim): The function with a single dim.
            x (Tensor): The tensor input.

        Returns:
            Tensor: The tensor output.
        """
        return unflatten_dims(
            func(flatten_dims(x, dims=self.dims), dim=self.dims[0]), dims=self.dims, shape=x.shape
        )


Ts = TypeVarTuple("Ts")


# TODO: for now the solution I found is using Generic[Unpack[TypeVarTuple]], but it does not bound
#       with Tuple[Tensor, ...], and extra cast is needed. Any better solution?
class TorchComposedParameter(AbstractTorchParameter, Generic[Unpack[Ts]]):
    """The base class for composed reparameterization."""

    def __init__(
        self,
        *params: AbstractTorchParameter,
        func: Callable[[Unpack[Ts]], Tensor],
        inv_func: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> None:
        """Init class.

        Args:
            *params (AbstractTorchParameter): The input param(s) to be composed.
            func (Callable[[*Ts], Tensor]): The function to compose the output from the parameters \
                given by the input param(s).
            inv_func (Optional[Callable[[Tensor], Tensor]], optional): The inverse of func (in \
                unary form), used to transform the intialization. For n-ary func, this inverse may \
                not be well-defined and is therefore NOT used. Defaults to lambda x: x.
        """
        super().__init__()
        # TODO: make ModuleList a generic?
        # ANNOTATE: We use List[Reparameterization] for typing so that elements are properly typed.
        # IGNORE: We must use nn.ModuleList for runtime to register sub-modules.
        self.params: List[AbstractTorchParameter] = nn.ModuleList(  # type: ignore[assignment]
            params
        )
        self.func = func
        self.inv_func = inv_func

    @cached_property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""
        dtype = self.params[0].dtype
        assert all(
            param.dtype == dtype for param in self.params
        ), "The dtype of all composing parameters should be the same."
        return dtype

    @property
    def device(self) -> torch.device:
        """The device of the output parameter."""
        device = self.params[0].device
        assert all(
            param.device == device for param in self.params
        ), "The device of all composing parameters should be the same."
        return device

    @torch.no_grad()
    def initialize(self, initializer_: Callable[[Tensor], Tensor]) -> None:
        """Initialize the internal parameter tensor(s) with the given initializer.

        This can only be called after materialization and will always overwrite whatever is \
        already in the internal param. To safely provide an initial value to a possibly reused \
        param, initialize through materialize() instead.

        The provided initializer_ is expected to provide an initial value for the output \
        parameter, and the value will be transformed through inv_func (as defined in __init__) and \
        passed to the input param, if there's only one input.
        In case of multiple inputs, the inputs are expected to initialized beforehand, as also in \
        materialization, and this function will silently become a no-op. This is because it's \
        generally difficult to define the inverse of n-ary functions and we just skip it for safety.

        Args:
            initializer_ (Callable[[Tensor], Tensor]): The function that initialize a Tensor \
                inplace while also returning the value.
        """
        if len(self.params) > 1:
            return

        # Construct a tmp Tensor -> initialize it -> transform by inv_func -> fill into x.
        self.params[0].initialize(
            lambda x: x.copy_(self.inv_func(initializer_(x.new_empty(self.shape))))
        )

    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # NOTE: params is not tuple, but generator still can be unpacked with *.
        # CAST: Expected Ts but got tuple (actually generator) of Tensor.
        params = cast(Tuple[Unpack[Ts]], (p() for p in self.params))
        return self.func(*params)


class TorchUnaryOpParameter(TorchComposedParameter[AbstractTorchParameter]):
    """The base class for unary composed reparameterization."""

    def __init__(
        self,
        param: AbstractTorchParameter,
        /,
        *,
        func: Callable[[Tensor], Tensor],
        inv_func: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> None:
        """Init class.

        Args:
            param (Optional[Reparameterization], optional): The input param to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
            func (Callable[[Tensor], Tensor]): The function to compose the output from the \
                parameters given by the input param.
            inv_func (Optional[Callable[[Tensor], Tensor]], optional): The inverse of func, used \
                to transform the intialization. Defaults to lambda x: x.
        """
        super().__init__(param, func=func)


class TorchBinaryOpParameter(
    TorchComposedParameter[AbstractTorchParameter, AbstractTorchParameter]
):
    """The base class for binary composed reparameterization."""

    def __init__(
        self,
        param1: AbstractTorchParameter,
        param2: AbstractTorchParameter,
        /,
        *,
        func: Callable[[Tensor, Tensor], Tensor],
        inv_func: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        """Init class.

        Args:
            param1 (AbstractTorchParameter): The first input param to be composed.
            param2 (AbstractTorchParameter): The second input param to be composed.
            func (Callable[[Tensor, Tensor], Tensor]): The function to compose the output from the \
                parameters given by the input reparams.
            inv_func (Optional[Callable[[Tensor], Tensor]], optional): Ignored. BinaryReparam does \
                not propagate initialization. Defaults to None.
        """
        super().__init__(param1, param2, func=func)


class TorchHadamardParameter(TorchBinaryOpParameter):
    """Hadamard product reparameterization."""

    def __init__(self, param1: AbstractTorchParameter, param2: AbstractTorchParameter) -> None:
        """Init class.

        Args:
            param1 (AbstractTorchParameter): The first input param to be composed.
            param2 (AbstractTorchParameter): The second input param to be composed.
        """
        assert param1.shape == param2.shape
        super().__init__(param1, param2, func=TorchHadamardParameter._forward)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.params[0].shape

    @staticmethod
    def _forward(x1: Tensor, x2: Tensor) -> Tensor:
        ...


class TorchKroneckerParameter(TorchBinaryOpParameter):
    """Kronecker product reparameterization."""

    def __init__(self, param1: AbstractTorchParameter, param2: AbstractTorchParameter) -> None:
        """Init class.

        Args:
            param1 (AbstractTorchParameter): The first input param to be composed.
            param2 (AbstractTorchParameter): The second input param to be composed.
        """
        super().__init__(param1, param2, func=torch.kron)


class TorchAffineParameter(TorchUnaryOpParameter):
    """Linear reparameterization.

    Range: (-inf, +inf), when a != 0.
    """

    def __init__(self, param: AbstractTorchParameter, /, *, a: float = 1, b: float = 0) -> None:
        """Init class.

        Args:
            param (Optional[Reparameterization], optional): The input param to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
            a (float, optional): The slope for the linear function. Defaults to 1.
            b (float, optional): The intercept for the linear function. Defaults to 0.
        """
        # Faster code path for simpler cases, to save some computations.
        # ANNOTATE: Specify signature for lambda.
        func: Callable[[Tensor], Tensor]
        inv_func: Callable[[Tensor], Tensor]
        # DISABLE: It's intended to use lambda here because it's too simple to use def.
        # pylint: disable=unnecessary-lambda-assignment
        # DISABLE: It's intended to explicitly compare with 0, so that it's easier to understand.
        # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
        if a == 1 and b == 0:
            func = inv_func = lambda x: x
        elif a == 1:  # and b != 0
            func = lambda x: x + b
            inv_func = lambda x: x - b
        elif b == 0:  # and a != 1
            func = lambda x: a * x
            inv_func = lambda x: (1 / a) * x
        else:  # a != 1 and b != 0
            func = lambda x: a * x + b  # TODO: possible FMA? addcmul?
            inv_func = lambda x: (1 / a) * (x - b)
        # pylint: enable=use-implicit-booleaness-not-comparison-to-zero
        # pylint: enable=unnecessary-lambda-assignment
        super().__init__(param, func=func, inv_func=inv_func)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.params[0].shape


class TorchExpParameter(TorchUnaryOpParameter):
    """Exp reparameterization.

    Range: (0, +inf).
    """

    def __init__(self, param: AbstractTorchParameter) -> None:
        """Init class.

        Args:
            param (Optional[Reparameterization], optional): The input param to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
        """
        super().__init__(param, func=torch.exp, inv_func=torch.log)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.params[0].shape


class TorchLogParameter(TorchUnaryOpParameter):
    """Log reparameterization.

    Range: (-inf, +inf).
    """

    def __init__(self, param: AbstractTorchParameter) -> None:
        """Init class.

        Args:
            param (Optional[Reparameterization], optional): The input param to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
        """
        super().__init__(param, func=torch.log, inv_func=torch.exp)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.params[0].shape


class TorchSquareParameter(TorchUnaryOpParameter):
    """Square reparameterization.

    Range: [0, +inf).
    """

    def __init__(self, param: AbstractTorchParameter) -> None:
        """Init class.

        Args:
            param (Optional[Reparameterization], optional): The input param to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
        """
        super().__init__(param, func=torch.square, inv_func=torch.sqrt)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.params[0].shape


class TorchClampParameter(TorchUnaryOpParameter):
    """Clamp reparameterization.

    Range: [min, max], as provided.
    """

    # DISABLE: We must use min/max as names, because this is the API of pytorch.
    def __init__(
        self,
        param: AbstractTorchParameter,
        /,
        *,
        min: Optional[float] = None,  # pylint: disable=redefined-builtin
        max: Optional[float] = None,  # pylint: disable=redefined-builtin
    ) -> None:
        """Init class.

        Args:
            param (Optional[Reparameterization], optional): The input param to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
            min (Optional[float], optional): The lower-bound for clamping, None for no clamping in \
                this direction. Defaults to None.
            max (Optional[float], optional): The upper-bound for clamping, None for no clamping in \
                this direction. Defaults to None.
        """
        # We assume the inv of clamp is identity.
        super().__init__(param, func=functools.partial(torch.clamp, min=min, max=max))

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.params[0].shape


class TorchSigmoidParameter(TorchUnaryOpParameter):
    """Sigmoid reparameterization.

    Range: (0, 1).
    """

    def __init__(self, param: AbstractTorchParameter) -> None:
        """Init class.

        Args:
            param (Optional[Reparameterization], optional): The input param to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
        """
        super().__init__(param, func=torch.sigmoid, inv_func=torch.logit)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.params[0].shape


class TorchReduceOpParameter(TorchUnaryOpParameter, _ApplySingleDimFuncMixin):
    """The base class for normalized reparameterization."""

    # NOTE: This class only serves as the common base of all normalized reparams, but include
    #       nothing more. It's up to the implementations to define further details.


class TorchSoftmaxParameter(TorchReduceOpParameter):
    """Softmax reparameterization.

    Range: (0, 1), 0 available if input is masked, 1 available when only one element valid.
    Constraints: sum to 1.
    """

    def __init__(self, param: AbstractTorchParameter) -> None:
        """Init class.

        Args:
            param (Optional[Reparameterization], optional): The input param to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
        """
        # Softmax is just scaled exp, so we take log as inv.
        super().__init__(param, func=self._func, inv_func=torch.log)

    def _func(self, x: Tensor) -> Tensor:
        # torch.softmax only accepts a single dim, so we need the applier.
        x = self._apply_single_dim_func(torch.softmax, x)
        # nan will appear when there's only one element and it's masked. In that case we project nan
        # as 1 (0 in log-sapce). +inf and -inf are (unsafely) projected but will not appear.
        return torch.nan_to_num(x, nan=1)


class TorchLogSoftmaxParameter(TorchReduceOpParameter, _ApplySingleDimFuncMixin):
    """LogSoftmax reparameterization, which is more numarically-stable than log(softmax(...)).

    Range: (-inf, 0), -inf available if input is masked, 0 available when only one element valid.
    Constraints: logsumexp to 0.
    """

    def __init__(self, param: Optional[AbstractTorchParameter] = None, /) -> None:
        """Init class.

        Args:
            param (Optional[Reparameterization], optional): The input param to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
        """
        # LogSoftmax is just additive offset, so we take identity as inv.
        super().__init__(param, func=self._func)

    def _func(self, x: Tensor) -> Tensor:
        # torch.log_softmax only accepts a single dim, so we need the applier.
        x = self._apply_single_dim_func(torch.log_softmax, x)
        # -inf still passes gradients, so we use a redundant projection to stop it. nan is the same
        # as SoftmaxReparam, projected to 0 (in log-space). +inf will not appear.
        return torch.nan_to_num(x, nan=0, neginf=float("-inf"))


class TorchScaledSigmoidParameter(TorchUnaryOpParameter):
    """Reparameterization for ExpFamily-Normal."""

    def __init__(
        self,
        param: AbstractTorchParameter,
        /,
        *,
        vmin: float = 0.0,
        vmax: float = 1.0,
    ) -> None:
        """Init class.

        Args:
            param (Optional[Reparameterization], optional): The input reparam to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
            vmin (float, optional): The min variance. Defaults to 0.0001.
            vmax (float, optional): The max variance. Defaults to 10.0.
        """
        # We don't assign inv_func and simply pass through the initialization.
        super().__init__(param, func=self._func)
        assert 0 <= vmin < vmax, "Must provide 0 <= min_var < max_var."
        self.vmin = vmin
        self.vmax = vmax

    def _func(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x[..., 1, :]) * (self.vmax - self.vmin) + self.vmin
