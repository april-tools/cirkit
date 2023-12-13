# pylint: disable=too-few-public-methods
# Disable: For this file we disable the above because all classes trigger this but it's intended.

import functools
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from cirkit.new.reparams.composed import ComposedReparam
from cirkit.new.reparams.reparam import Reparameterization


class UnaryReparam(ComposedReparam[Tensor]):
    """The unary composed reparameterization."""

    # TODO: pylint is wrong?
    # Disable: This is not useless as the signature of __init__ has changed.
    def __init__(  # pylint: disable=useless-parent-delegation
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
        func: Callable[[Tensor], Tensor]
        inv_func: Callable[[Tensor], Tensor]
        # Disable: It's intended to use lambda here -- too simple to use def.
        # pylint: disable=unnecessary-lambda-assignment
        # Disable: It's intended to explicitly compare with 0, so that it's easier to understand.
        # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
        if a == 1 and b == 0:
            func = inv_func = lambda x: x
        elif a == 1:  # and b != 0
            func = lambda x: x + b
            inv_func = lambda x: x - b
        elif b == 0:  # and a != 1
            func = lambda x: a * x
            inv_func = lambda x: (1 / a) * x
        else:
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

    # Disable: We must use min/max as names, so that it's in line with pytorch.
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
            min (Optional[float], optional): The lower-bound for clamping, None to disable this \
                direction. Defaults to None.
            max (Optional[float], optional): The upper-bound for clamping, None to disable this \
                direction. Defaults to None.
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
