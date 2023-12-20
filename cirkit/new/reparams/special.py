# pylint: disable=too-few-public-methods
# DISABLE: For this file we disable the above because all classes trigger it and it's intended.

from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from cirkit.new.reparams.reparam import Reparameterization
from cirkit.new.reparams.unary import UnaryReparam

# This file is for specialized reparams designed for one single purpose, e.g., specifically for one
# InputLayer implementation. unary.py/normalized.py shoud be preferred for general reparams.


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
        var = (
            torch.sigmoid(x[..., 1, :]) * (self.max_var - self.min_var) + self.min_var
        )  # shape (..., :).
        param = torch.stack(
            (mu, torch.tensor(-0.5).to(mu).expand_as(mu)), dim=-2
        )  # shape (..., 2, :).
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
