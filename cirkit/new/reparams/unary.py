# pylint: disable=too-few-public-methods
# DISABLE: For this file we disable the above because all classes trigger it and it's intended.

import functools
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from cirkit.new.reparams.composed import ComposedReparam
from cirkit.new.reparams.reparam import Reparameterization


class UnaryReparam(ComposedReparam[Tensor]):
    """The base class for unary composed reparameterization."""

    # TODO: pylint is wrong?
    # DISABLE: This is not useless as the signature of __init__ has changed.
    # pylint: disable-next=useless-parent-delegation
    def __init__(
        self,
        reparam: Optional[Reparameterization] = None,
        /,
        *,
        func: Callable[[Tensor], Tensor],
        inv_func: Optional[Callable[[Tensor], Union[Tuple[Tensor], Tensor]]] = None,
    ) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            func (Callable[[Tensor], Tensor]): The function to compose the output from the \
                parameters given by reparam.
            inv_func (Optional[Callable[[Tensor], Union[Tuple[Tensor], Tensor]]], optional): The \
                inverse of func, used to transform the intialization. The initializer will \
                directly pass through if no inv_func provided. Defaults to None.
        """
        super().__init__(reparam, func=func, inv_func=inv_func)


# TODO: how do we annotate the range? or use some way to calc and propagate it?


class LogMaskReparam(UnaryReparam):
    """Mask (in log-space) reparameterization.

    Range: input unchanged, if not masked; -inf, at masked position.
    """

    log_mask: Optional[Tensor]
    """The log of normalization mask, shape same as the parameter itself."""

    def __init__(
        self,
        reparam: Optional[Reparameterization] = None,
        /,
        *,
        mask: Optional[Tensor] = None,
        log_mask: Optional[Tensor] = None,
    ) -> None:
        """Init class.

        At most one of mask and log_mask may be provided. Masking will be skipped if neither is \
        provided or the mask is full (nothing masked out).

        Args:
            reparam (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            mask (Optional[Tensor], optional): The 0/1 mask for valid positions. None for the \
                other or no masking. If not None, the shape must be broadcastable to the shape \
                used in materialization. Defaults to None.
            log_mask (Optional[Tensor], optional): The -inf/0 mask for valid positions. None for \
                the other or no masking. If not None, the shape must be broadcastable to the shape \
                used in materialization. Defaults to None.
        """
        assert mask is None or log_mask is None, "mask and log_mask may not be supplied together."

        # Broadcast is not checked now because we don't have self.shape.
        if mask is not None:
            log_mask = torch.log(mask)

        # A non-0, i.e., -inf, element means that position is masked out, so mask is not full.
        if log_mask is not None and log_mask.any():
            # TODO: check if it's ok to pass through -inf in inv?
            # We assume the inv is also masked.
            super().__init__(reparam, func=lambda x: x + log_mask, inv_func=lambda x: x + log_mask)
            self.register_buffer("log_mask", log_mask)  # register_* only work after __init__().
        else:
            super().__init__(reparam, func=lambda x: x)  # inv_func is identity by default.
            self.register_buffer("log_mask", None)

    def materialize(self, shape: Sequence[int], /, *, dim: Union[int, Sequence[int]]) -> bool:
        """Materialize the internal parameter tensors with given shape.

        If it is already materialized, False will be returned to indicate no materialization. \
        However, a second call to materialize must give the same config, so that the underlying \
        params can indeed be reused.

        The initial value of the parameter after materialization is not guaranteed, and explicit \
        initialization is expected.

        The kwarg, dim, is used to hint the normalization of sum weights. It's not always used but \
        must be supplied with the sum-to-1 dimension(s) so that it's guaranteed to be available \
        when a normalized reparam is passed as self.

        Args:
            shape (Sequence[int]): The shape of the output parameter.
            dim (Union[int, Sequence[int]]): The dimension(s) along which the normalization will \
                be applied. However a subclass impl may choose to ignore this.

        Returns:
            bool: Whether the materialization is done.
        """
        if self.log_mask is not None:
            # An easy way to check if broadcastable: broadcast_to raises RuntimeError when not.
            self.log_mask.broadcast_to(shape)
        # Here we only check shape broadcast. Delegate everything else to super().
        return super().materialize(shape, dim=dim)


class LinearReparam(UnaryReparam):
    """Linear reparameterization.

    Range: (-inf, +inf), when a != 0.
    """

    def __init__(
        self, reparam: Optional[Reparameterization] = None, /, *, a: float = 1, b: float = 0
    ) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
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
            func = lambda x: a * x + b  # TODO: possible FMA?
            inv_func = lambda x: (1 / a) * (x - b)
        # pylint: enable=use-implicit-booleaness-not-comparison-to-zero
        # pylint: enable=unnecessary-lambda-assignment
        super().__init__(reparam, func=func, inv_func=inv_func)


class ExpReparam(UnaryReparam):
    """Exp reparameterization.

    Range: (0, +inf).
    """

    def __init__(self, reparam: Optional[Reparameterization] = None, /) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
        """
        super().__init__(reparam, func=torch.exp, inv_func=torch.log)


class SquareReparam(UnaryReparam):
    """Square reparameterization.

    Range: [0, +inf).
    """

    def __init__(self, reparam: Optional[Reparameterization] = None, /) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
        """
        super().__init__(reparam, func=torch.square, inv_func=torch.sqrt)


class ClampReparam(UnaryReparam):
    """Clamp reparameterization.

    Range: [min, max], as provided.
    """

    # DISABLE: We must use min/max as names, because this is the API of pytorch.
    def __init__(
        self,
        reparam: Optional[Reparameterization] = None,
        /,
        *,
        min: Optional[float] = None,  # pylint: disable=redefined-builtin
        max: Optional[float] = None,  # pylint: disable=redefined-builtin
    ) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            min (Optional[float], optional): The lower-bound for clamping, None for no clamping in \
                this direction. Defaults to None.
            max (Optional[float], optional): The upper-bound for clamping, None for no clamping in \
                this direction. Defaults to None.
        """
        # We assume the inv of clamp is identity.
        super().__init__(reparam, func=functools.partial(torch.clamp, min=min, max=max))


class SigmoidReparam(UnaryReparam):
    """Sigmoid reparameterization.

    Range: (0, 1).
    """

    def __init__(self, reparam: Optional[Reparameterization] = None, /) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
        """
        super().__init__(reparam, func=torch.sigmoid, inv_func=torch.logit)
