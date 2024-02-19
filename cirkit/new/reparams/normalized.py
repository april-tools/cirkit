# pylint: disable=too-few-public-methods
# DISABLE: For this file we disable the above because all classes trigger it and it's intended.

from typing import Optional, Protocol, Tuple

import torch
from torch import Tensor

from cirkit.new.reparams.reparam import Reparameterization
from cirkit.new.reparams.unary import UnaryReparam
from cirkit.new.utils import flatten_dims, unflatten_dims

# This file is for and only for unary reparams that include normalization.


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


class NormalizedReparam(UnaryReparam):
    """The base class for normalized reparameterization."""

    # NOTE: This class only serves as the common base of all normalized reparams, but include
    #       nothing more. It's up to the implementations to define further details.


class SoftmaxReparam(NormalizedReparam, _ApplySingleDimFuncMixin):
    """Softmax reparameterization.

    Range: (0, 1), 0 available if input is masked, 1 available when only one element valid.
    Constraints: sum to 1.
    """

    def __init__(self, reparam: Optional[Reparameterization] = None, /) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparam to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
        """
        # Softmax is just scaled exp, so we take log as inv.
        super().__init__(reparam, func=self._func, inv_func=torch.log)

    def _func(self, x: Tensor) -> Tensor:
        # torch.softmax only accepts a single dim, so we need the applier.
        x = self._apply_single_dim_func(torch.softmax, x)
        # nan will appear when there's only one element and it's masked. In that case we project nan
        # as 1 (0 in log-sapce). +inf and -inf are (unsafely) projected but will not appear.
        return torch.nan_to_num(x, nan=1)


class LogSoftmaxReparam(NormalizedReparam, _ApplySingleDimFuncMixin):
    """LogSoftmax reparameterization, which is more numarically-stable than log(softmax(...)).

    Range: (-inf, 0), -inf available if input is masked, 0 available when only one element valid.
    Constraints: logsumexp to 0.
    """

    def __init__(self, reparam: Optional[Reparameterization] = None, /) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparam to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
        """
        # LogSoftmax is just additive offset, so we take identity as inv.
        super().__init__(reparam, func=self._func)

    def _func(self, x: Tensor) -> Tensor:
        # torch.log_softmax only accepts a single dim, so we need the applier.
        x = self._apply_single_dim_func(torch.log_softmax, x)
        # -inf still passes gradients, so we use a redundant projection to stop it. nan is the same
        # as SoftmaxReparam, projected to 0 (in log-space). +inf will not appear.
        return torch.nan_to_num(x, nan=0, neginf=float("-inf"))
