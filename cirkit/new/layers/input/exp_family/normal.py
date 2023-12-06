import math
from typing import Literal, cast

import torch
from torch import Tensor

from cirkit.new.layers.input.exp_family.exp_family import ExpFamilyLayer
from cirkit.new.reparams import Reparameterization


class NormalLayer(ExpFamilyLayer):
    """The Normal distribution layer.

    This is fully factorized down to univariate Categorical distributions.
    """

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: Literal[1] = 1,
        reparam: Reparameterization,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (Literal[1], optional): The arity of the layer, must be 1. Defaults to 1.
            reparam (Reparameterization): The reparameterization for layer parameters. Expected to \
                be EFNormalReparam.
            num_categories (int): The number of categories for Categorical distribution.
        """
        self.suff_stats_shape = (2, num_input_units)  # 2 for mean and var.
        # Set self.suff_stats_shape before ExpFamilyLayer.__init__. Although dim=-1 is marked to
        # normalize there, the reparam is expected to be EFNormalReparam which ignores the dim.
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        # log_h is a fixed value, so calc in __init__ instead of on the fly.
        self._log_h = -0.5 * self.num_input_units * math.log(2 * math.pi)

    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, K).

        Returns:
            Tensor: The sufficient statistics T, shape (*B, H, S).
        """
        # TODO: torch __pow__ issue
        return torch.cat((x, cast(Tensor, x**2)), dim=-1).movedim(
            0, -2
        )  # shape (H, *B, 2*K) -> (*B, H, S=2*K).

    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, K).

        Returns:
            Tensor: The natural parameters eta, shape (*B, H).
        """
        return torch.tensor(self._log_h).to(x).expand_as(x[..., 0]).movedim(0, -1)

    def log_partition(self, eta: Tensor) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (H, K, *S).

        Returns:
            Tensor: The log partition function A, shape (H, K).
        """
        eta1 = eta[..., 0, :]  # shape (H, K, Ki).
        eta2 = eta[..., 1, :]  # shape (H, K, Ki).
        # TODO: torch __pow__ issue
        log_normalizer = -0.25 * cast(Tensor, eta1**2) / eta2 - 0.5 * torch.log(-2 * eta2)
        return log_normalizer.sum(dim=-1)  # shape (H, K).
