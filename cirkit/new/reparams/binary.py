from typing import Callable, Optional, Tuple, Union

from torch import Tensor

from cirkit.new.reparams.composed import ComposedReparam
from cirkit.new.reparams.reparam import Reparameterization


class BinaryReparam(ComposedReparam[Tensor, Tensor]):
    """The binary composed reparameterization."""

    def __init__(
        self,
        reparam1: Optional[Reparameterization] = None,
        reparam2: Optional[Reparameterization] = None,
        /,
        *,
        func: Callable[[Tensor, Tensor], Tensor],
        inv_func: Optional[Callable[[Tensor], Union[Tuple[Tensor, Tensor], Tensor]]] = None,
    ) -> None:
        # pylint: disable=line-too-long  # Disable: This long line is unavoidable.
        """Init class.

        Args:
            reparam1 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            reparam2 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            func (Callable[[Tensor, Tensor], Tensor]): The function to compose the output from the \
                parameters given by reparam.
            inv_func (Optional[Callable[[Tensor], Union[Tuple[Tensor, Tensor], Tensor]]], optional): \
                The inverse of func, used to transform the intialization. The initializer will \
                directly pass through if no inv_func provided. Defaults to None.
        """
        # pylint: enable=line-too-long
        super().__init__(reparam1, reparam2, func=func, inv_func=inv_func)


# TODO: circuit product
