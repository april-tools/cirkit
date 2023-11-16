import math
from typing import Literal, cast

import torch
from torch import Tensor

from cirkit.reparams.exp_family import ReparamEFNormal
from cirkit.utils.type_aliases import ReparamFactory

from .exp_family import ExpFamilyLayer


class NormalLayer(ExpFamilyLayer):
    """The normal distribution layer."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_vars: int,
        num_channels: int = 1,
        num_replicas: int = 1,
        num_input_units: Literal[1] = 1,
        num_output_units: int,
        arity: Literal[1] = 1,
        num_folds: Literal[0] = 0,
        fold_mask: None = None,
        reparam: ReparamFactory = ReparamEFNormal,
    ) -> None:
        """Init class.

        Args:
            num_vars (int): The number of variables of the circuit.
            num_channels (int, optional): The number of channels of each variable. Defaults to 1.
            num_replicas (int, optional): The number of replicas for each variable. Defaults to 1.
            num_input_units (Literal[1], optional): The number of input units, must be 1. \
                Defaults to 1.
            num_output_units (int): The number of output units.
            arity (Literal[1], optional): The arity of the layer, must be 1. Defaults to 1.
            num_folds (Literal[0], optional): The number of folds. Should not be provided and will \
                be calculated as num_vars*num_replicas. Defaults to 0.
            fold_mask (None, optional): The mask of valid folds, must be None. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamEFNormal.
        """
        super().__init__(
            num_vars=num_vars,
            num_channels=num_channels,
            num_replicas=num_replicas,
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam,
            num_suff_stats=2 * num_channels,
        )
        self._log_h = -0.5 * self.num_channels * math.log(2 * math.pi)

    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (*B, D, C).

        Returns:
            Tensor: The sufficient statistics T, shape (*B, D, S).
        """
        # TODO: torch __pow__ issue
        return torch.cat((x, x**2), dim=-1)  # type: ignore[misc]  # shape (*B, D, S=2*C)

    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (*B, D, C).

        Returns:
            Tensor: The natural parameters eta, shape (*B, D).
        """
        return torch.tensor(self._log_h).to(x).expand_as(x[..., 0])

    def log_partition(self, eta: Tensor) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (D, K, P, S).

        Returns:
            Tensor: The log partition function A, shape (D, K, P).
        """
        eta1 = eta[..., : self.num_channels]  # shape (D, K, P, C)
        eta2 = eta[..., self.num_channels :]  # shape (D, K, P, C)
        # TODO: torch __pow__ issue
        log_normalizer = -0.25 * cast(Tensor, eta1**2) / (eta2) - 0.5 * torch.log(-2 * eta2)
        return log_normalizer.sum(dim=-1)  # shape (D, K, P)

    @property
    def mean(self) -> Tensor:
        """The parameter mu (mean) for normal distribution, shape (D, K, P, C)."""
        param = self.params()
        return -0.5 * param[..., : self.num_channels] / param[..., self.num_channels :]

    @property
    def variance(self) -> Tensor:
        """The parameter sigma^2 (variance) for normal distribution, shape (D, K, P, C)."""
        param = self.params()
        # TODO: pytorch __rdiv__ issue
        return -0.5 / param[..., self.num_channels :]  # type: ignore[no-any-return,misc]
