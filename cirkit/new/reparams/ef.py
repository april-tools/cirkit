# pylint: disable=too-few-public-methods
# DISABLE: For this file we disable the above because all classes trigger it and it's intended.

from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from cirkit.new.reparams.binary import BinaryReparam
from cirkit.new.reparams.reparam import Reparameterization
from cirkit.new.reparams.unary import UnaryReparam

# This file is for specialized reparams designed specifically for ExpFamilyLayer.


class EFNormalReparam(UnaryReparam):
    """Reparameterization for ExpFamily-Normal."""

    def __init__(
        self,
        reparam: Optional[Reparameterization] = None,
        /,
        *,
        min_var: float = 0.0001,
        max_var: float = 10.0,
    ) -> None:
        """Init class.

        Args:
            reparam (Optional[Reparameterization], optional): The input reparameterization to be \
                composed. If None, a LeafReparam will be constructed in its place. Defaults to None.
            min_var (float, optional): The min variance. Defaults to 0.0001.
            max_var (float, optional): The max variance. Defaults to 10.0.
        """
        # We don't assign inv_func and simply pass through the initialization.
        super().__init__(reparam, func=self._func)

        assert 0 <= min_var < max_var, "Must provide 0 <= min_var < max_var."
        self.min_var = min_var
        self.max_var = max_var

    def _func(self, x: Tensor) -> Tensor:
        # In materialize, shape[-2] == 2 is asserted.
        mu = x[..., 0, :]  # shape (..., :).
        # shape (..., :).
        var = torch.sigmoid(x[..., 1, :]) * (self.max_var - self.min_var) + self.min_var
        # shape (..., 2, :).
        param = torch.stack((mu, torch.tensor(-0.5).to(mu).expand_as(mu)), dim=-2)
        return param / var.unsqueeze(dim=-2)  # shape (..., 2, :).

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
        # TODO: how should this chain with mask? need to doc the usage. or do we need mask at all?
        assert shape[-2] == 2, "The shape does not fit the requirement of EF-Normal."
        return super().materialize(shape, dim=dim)


class EFProductReparam(BinaryReparam):
    """Reparameterization for product of Exponential Family.

    This is designed to do the "kronecker concat" with flattened suff stats:
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
        super().__init__(reparam1, reparam2, func=self._func)

    @staticmethod
    def _func(param1: Tensor, param2: Tensor) -> Tensor:
        # shape (H, K, *S) -> (H, K, S) -> (H, K, 1, S).
        param1 = param1.flatten(start_dim=2).unsqueeze(dim=2)
        # shape (H, K, *S) -> (H, K, S) -> (H, 1, K, S).
        param2 = param2.flatten(start_dim=2).unsqueeze(dim=1)
        # IGNORE: broadcast_tensors is not typed.
        # shape (H, K, K, S+S) -> (H, K*K, S+S).
        return torch.cat(
            torch.broadcast_tensors(param1, param2), dim=-1  # type: ignore[no-untyped-call,misc]
        ).flatten(start_dim=1, end_dim=2)
