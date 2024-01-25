# pylint: disable=too-few-public-methods
# DISABLE: For this file we disable the above because all classes trigger it and it's intended.

from typing import Optional, Protocol, Tuple

import torch
from torch import Tensor

from cirkit.new.reparams.reparam import Reparameterization
from cirkit.new.reparams.unary import UnaryReparam
from cirkit.new.utils import flatten_dims, unflatten_dims

# This file is for unary reparams that includes normalization. unary.py should be preferred for
# simple reparams.


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


class _NormalizedReparamMixin:
    """A mixin for helpers useful for reparams with normalization on some dims."""

    def _apply_normalizer(
        self: _HasDimsTuple, normalizer: _TensorFuncWithDim, x: Tensor, /
    ) -> Tensor:
        """Apply a normalizer function on a Tensor over self.dims.

        Args:
            normalizer (_TensorFuncWithDim): The normalizer of a tensor with a dim arg.
            x (Tensor): The tensor input.

        Returns:
            Tensor: The normalized output.
        """
        return unflatten_dims(
            normalizer(flatten_dims(x, dims=self.dims), dim=self.dims[0]),
            dims=self.dims,
            shape=x.shape,
        )


class SoftmaxReparam(UnaryReparam, _NormalizedReparamMixin):
    """Softmax reparameterization.

    Range: (0, 1), 0 available if input is masked, 1 available when only one element valid.
    Constraints: sum to 1.
    """

    def __init__(self, reparam: Optional[Reparameterization] = None, /) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
        """
        # Softmax is just scaled exp, so we take log as inv.
        super().__init__(reparam, func=self._func, inv_func=torch.log)

    def _func(self, x: Tensor) -> Tensor:
        # torch.softmax can only accept one dim, so we need to rearrange dims.
        x = self._apply_normalizer(torch.softmax, x)
        # nan will appear when there's only 1 element and it's masked. In that case we projecte nan
        # as 1 (0 in log-sapce). +inf and -inf are (unsafely) projected but will not appear.
        return torch.nan_to_num(x, nan=1)


class LogSoftmaxReparam(UnaryReparam, _NormalizedReparamMixin):
    """LogSoftmax reparameterization, which is more numarically-stable than log(softmax(...)).

    Range: (-inf, 0), -inf available if input is masked, 0 available when only one element valid.
    Constraints: logsumexp to 0.
    """

    def __init__(self, reparam: Optional[Reparameterization] = None, /) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
        """
        # Log_softmax is just an offset, so we take identity as inv.
        super().__init__(reparam, func=self._func)

    def _func(self, x: Tensor) -> Tensor:
        # torch.log_softmax can only accept one dim, so we need to rearrange dims.
        x = self._apply_normalizer(torch.log_softmax, x)
        # -inf still passes gradients, so we use a redundant projection to stop it. nan is the same
        # as SoftmaxReparam, projected to 0 (in log-space). +inf will not appear.
        return torch.nan_to_num(x, nan=0, neginf=float("-inf"))
