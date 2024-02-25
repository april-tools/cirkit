# pylint: disable=too-few-public-methods
# DISABLE: For this file we disable the above because all classes trigger it and it's intended.

import functools
from typing import Callable, Optional, Sequence, Union

import torch
from torch import Tensor

from cirkit.new.reparams.composed import ComposedReparam
from cirkit.new.reparams.leaf import LeafReparam
from cirkit.new.reparams.reparam import Reparameterization


class UnaryReparam(ComposedReparam[Tensor]):
    """The base class for unary composed reparameterization."""

    def __init__(
        self,
        reparam: Optional[Reparameterization] = None,
        /,
        *,
        func: Callable[[Tensor], Tensor],
        inv_func: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparam to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
            func (Callable[[Tensor], Tensor]): The function to compose the output from the \
                parameters given by the input reparam.
            inv_func (Optional[Callable[[Tensor], Tensor]], optional): The inverse of func, used \
                to transform the intialization. Defaults to lambda x: x.
        """
        super().__init__(
            reparam if reparam is not None else LeafReparam(), func=func, inv_func=inv_func
        )


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
            reparam (Optional[Reparameterization], optional): The input reparam to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
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

        # A non-0, i.e. -inf, element means that position is masked out, so mask is not full.
        if log_mask is not None and log_mask.any():
            # TODO: check if it's ok to pass through -inf in inv?
            # We assume the inv is also masked.
            super().__init__(reparam, func=lambda x: x + log_mask, inv_func=lambda x: x + log_mask)
        else:
            log_mask = None  # All-0, i.e. log(1), mask is not useful.
            super().__init__(reparam, func=lambda x: x)  # Saves a +0.

        self.register_buffer("log_mask", log_mask)  # register_* only work after __init__().

    def materialize(
        self,
        shape: Sequence[int],
        /,
        *,
        dim: Union[int, Sequence[int]],
        initializer_: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> bool:
        """Materialize the internal parameter tensor(s) with given shape and initialize if required.

        Materialization (and optionally initialization) is only executed if it's not materialized \
        yet. Otherwise this function will become a silent no-op, providing safe reuse of the same \
        reparam. However, the arguments must be the same among re-materialization attempts, to \
        make sure the reuse is consistent. The return value will indicate whether there's \
        materialization happening.

        The kwarg-only dim, is used to hint the normalization of sum weights (or some input params \
        that may expect normalization). It's not always used by all layers but is required to be\
        supplied with the sum-to-1 dimension(s) so that both normalized and unnormalized reparams \
        will work under the same materialization setting.

        If an initializer_ is provided, it will be used to fill the initial value of the "output" \
        parameter, and it will be masked to be passed to input. If no initializer is given, the \
        internal storage will contain random memory.

        Args:
            shape (Sequence[int]): The shape of the output parameter.
            dim (Union[int, Sequence[int]]): The dimension(s) along which the normalization will \
                be applied. Unnormalized implementations may choose to ignore this.
            initializer_ (Optional[Callable[[Tensor], Tensor]], optional): The function that \
                initialize a Tensor inplace while also returning the value. Leave default for no \
                initialization. Defaults to None.

        Returns:
            bool: Whether the materialization is actually performed.
        """
        if self.log_mask is not None:
            # An easy way to check if broadcastable: broadcast_to raises RuntimeError when not.
            self.log_mask.broadcast_to(shape)
        # Here we only check shape broadcast. Delegate everything else to super().
        return super().materialize(shape, dim=dim, initializer_=initializer_)


class LinearReparam(UnaryReparam):
    """Linear reparameterization.

    Range: (-inf, +inf), when a != 0.
    """

    def __init__(
        self, reparam: Optional[Reparameterization] = None, /, *, a: float = 1, b: float = 0
    ) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparam to be composed. If \
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
        super().__init__(reparam, func=func, inv_func=inv_func)


class ExpReparam(UnaryReparam):
    """Exp reparameterization.

    Range: (0, +inf).
    """

    def __init__(self, reparam: Optional[Reparameterization] = None, /) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparam to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
        """
        super().__init__(reparam, func=torch.exp, inv_func=torch.log)


class SquareReparam(UnaryReparam):
    """Square reparameterization.

    Range: [0, +inf).
    """

    def __init__(self, reparam: Optional[Reparameterization] = None, /) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparam to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
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
            reparam (Optional[Reparameterization], optional): The input reparam to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
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
            reparam (Optional[Reparameterization], optional): The input reparam to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
        """
        super().__init__(reparam, func=torch.sigmoid, inv_func=torch.logit)
