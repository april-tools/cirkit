from typing import Callable, List, Optional, Tuple, Union

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
        # DISABLE: This long line is unavoidable for Args doc.
        # pylint: disable=line-too-long
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


class ProductReparam(BinaryReparam):
    """reparameterization for the product of circuits."""

    def __init__(
        self,
        reparam1: Reparameterization,
        reparam2: Reparameterization,
        /,
    ) -> None:
        # DISABLE: This long line is unavoidable for Args doc.
        """Init class.

        Args:
            reparam1 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            reparam2 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
        """
        super().__init__(reparam1, reparam2, func=None, inv_func=None)  # type: ignore[arg-type]

        if reparam1.is_materialized and reparam2.is_materialized:
            self.shape = reparam1.shape
            self.dims = reparam1.dims  # TODO: product with different param shape

    def forward(self) -> List[Tensor]:  # type: ignore[override]
        """Get the reparameterized parameters.

        Returns:
            List[Tensor]: The list of parameters in the circuit product.
        """
        # pylint: disable=line-too-long
        # flatten nested lists into a single list
        params = [  # type: ignore[misc]
            rep  # type: ignore[misc]
            for reparam in self.reparams
            for rep in (reparam() if isinstance(reparam(), list) else [reparam()])  # type: ignore[attr-defined]
        ]
        return params  # type: ignore[misc]
