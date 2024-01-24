from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from cirkit.new.reparams.composed import ComposedReparam
from cirkit.new.reparams.reparam import Reparameterization


class BinaryReparam(ComposedReparam[Tensor, Tensor]):
    """The base class for binary composed reparameterization."""

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


class KroneckerReparam(BinaryReparam):
    """Reparameterization by kronecker product."""

    def __init__(
        self,
        reparam1: Optional[Reparameterization] = None,
        reparam2: Optional[Reparameterization] = None,
        /,
    ) -> None:
        """Init class.

        NOTE: Be careful about passing None for this reparam. It might be unexpected.

        Args:
            reparam1 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            reparam2 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
        """
        super().__init__(reparam1, reparam2, func=torch.kron, inv_func=None)


class EFProductReparam(BinaryReparam):
    """Reparameterization for product of Exponential Family.

    This is designed to do the "kronecker concat":
        - Expected input: (H, K_1, *S_1), (H, K_2, *S_2);
        - Will output: (H, K_1*K_2, flatten(S_1)+flatten(S_2)).
    """

    def __init__(
        self,
        reparam1: Optional[Reparameterization] = None,
        reparam2: Optional[Reparameterization] = None,
        /,
    ) -> None:
        """Init class.

        NOTE: Be careful about passing None for this reparam. It might be unexpected.

        Args:
            reparam1 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            reparam2 (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
        """
        super().__init__(reparam1, reparam2, func=self._func, inv_func=None)

    @staticmethod
    def _func(param1: Tensor, param2: Tensor) -> Tensor:
        param1 = param1.flatten(start_dim=2).unsqueeze(
            dim=2
        )  # shape (H, K, *S) -> (H, K, S) -> (H, K, 1, S).
        param2 = param2.flatten(start_dim=2).unsqueeze(
            dim=1
        )  # shape (H, K, *S) -> (H, K, S) -> (H, 1, K, S).
        # IGNORE: broadcast_tensors is not typed.
        return torch.cat(
            torch.broadcast_tensors(param1, param2), dim=-1  # type: ignore[no-untyped-call,misc]
        ).flatten(
            start_dim=1, end_dim=2
        )  # shape (H, K, K, S+S) -> (H, K*K, S+S).
