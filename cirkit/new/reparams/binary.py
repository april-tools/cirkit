from typing import Callable, Optional

import torch
from torch import Tensor

from cirkit.new.reparams.composed import ComposedReparam
from cirkit.new.reparams.reparam import Reparameterization


class BinaryReparam(ComposedReparam[Tensor, Tensor]):
    """The base class for binary composed reparameterization."""

    def __init__(
        self,
        reparam1: Reparameterization,
        reparam2: Reparameterization,
        /,
        *,
        func: Callable[[Tensor, Tensor], Tensor],
        inv_func: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        """Init class.

        Args:
            reparam1 (Reparameterization): The first input reparam to be composed.
            reparam2 (Reparameterization): The second input reparam to be composed.
            func (Callable[[Tensor, Tensor], Tensor]): The function to compose the output from the \
                parameters given by the input reparams.
            inv_func (Optional[Callable[[Tensor], Tensor]], optional): Ignored. BinaryReparam does \
                not propagate initialization. Defaults to None.
        """
        super().__init__(reparam1, reparam2, func=func)


class KroneckerReparam(BinaryReparam):
    """Kronecker product reparameterization."""

    def __init__(
        self,
        reparam1: Reparameterization,
        reparam2: Reparameterization,
        /,
    ) -> None:
        """Init class.

        Args:
            reparam1 (Reparameterization): The first input reparam to be composed.
            reparam2 (Reparameterization): The second input reparam to be composed.
        """
        super().__init__(reparam1, reparam2, func=torch.kron)
