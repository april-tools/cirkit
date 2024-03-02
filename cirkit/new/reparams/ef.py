# pylint: disable=too-few-public-methods
# DISABLE: For this file we disable the above because all classes trigger it and it's intended.

from typing import Callable, Optional, Sequence, Union

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
            reparam (Optional[Reparameterization], optional): The input reparam to be composed. If \
                None, a LeafReparam will be automatically constructed. Defaults to None.
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
        param = torch.stack((mu, mu.new_full((), -0.5).expand_as(mu)), dim=-2)
        return param / var.unsqueeze(dim=-2)  # shape (..., 2, :).

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

        The normalization dim is ignored because Normal distributin is not normalized through the \
        params but through its own partition function.

        If an initializer_ is provided, it will be used to fill the initial value. If no \
        initializer is given, the internal storage will contain random memory.

        Args:
            shape (Sequence[int]): The shape of the output parameter.
            dim (Union[int, Sequence[int]]): Ignored. This reparam is not normalized.
            initializer_ (Optional[Callable[[Tensor], Tensor]], optional): The function that \
                initialize a Tensor inplace while also returning the value. Leave default for no \
                initialization. Defaults to None.

        Returns:
            bool: Whether the materialization is actually performed.
        """
        # TODO: how should this chain with mask? need to doc the usage. or do we need mask at all?
        assert shape[-2] == 2, "The shape does not fit the requirement of EF-Normal."
        # Check shape and delegate everything else to super().
        return super().materialize(shape, dim=dim, initializer_=initializer_)


class EFProductReparam(BinaryReparam):
    """Reparameterization for product of Exponential Family.

    This is designed to do the "kronecker concat" with flattened suff stats:
        - Expected input: (H, K_1, *S_1), (H, K_2, *S_2);
        - Will output: (H, K_1*K_2, flatten(S_1)+flatten(S_2)).
    """

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
        super().__init__(reparam1, reparam2, func=self._func)

    @classmethod
    def _func(cls, param1: Tensor, param2: Tensor) -> Tensor:
        # shape (H, K, *S) -> (H, K, S) -> (H, K, 1, S).
        param1 = param1.flatten(start_dim=2).unsqueeze(dim=2)
        # shape (H, K, *S) -> (H, K, S) -> (H, 1, K, S).
        param2 = param2.flatten(start_dim=2).unsqueeze(dim=1)
        # IGNORE: broadcast_tensors is not typed.
        # shape (H, K, K, S+S) -> (H, K*K, S+S).
        return torch.cat(
            torch.broadcast_tensors(param1, param2), dim=-1  # type: ignore[no-untyped-call,misc]
        ).flatten(start_dim=1, end_dim=2)
