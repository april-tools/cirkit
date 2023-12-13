# pylint: disable=too-few-public-methods
# Disable: For this file we disable the above because all classes trigger this but it's intended.

from typing import Optional, Sequence
from typing_extensions import Unpack  # TODO: in typing from 3.12 for Unpack[dict]

import torch
from torch import Tensor

from cirkit.new.reparams.reparam import Reparameterization
from cirkit.new.reparams.unary import UnaryReparam
from cirkit.new.utils.type_aliases import MaterializeKwargs

# This file is for some specialized reparams designed one purpose, e.g., specifically for one
# InputLayer implementation. Simple reparams should still prefer to fit into unary.py/normalized.py.


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

    def materialize(self, shape: Sequence[int], /, **kwargs: Unpack[MaterializeKwargs]) -> bool:
        """Materialize the internal parameter tensors with given shape.

        If it is already materialized, False will be returned to indicate no materialization. \
        However, a second call to materialize must give the same config, so that the underlying \
        params can indeed be reused.

        The initial value of the parameter after materialization is not guaranteed, and explicit \
        initialization is expected.

        Args:
            shape (Sequence[int]): The shape of the output parameter.
            **kwargs (Unpack[MaterializeKwargs]): Passed to super().materialize().

        Returns:
            bool: Whether the materialization is done.
        """
        # TODO: do we use mask? or do we need mask at all?
        assert shape[-2] == 2, "The shape does not fit the requirement of EF-Normal."
        return super().materialize(shape, **kwargs)
