import math
from typing import cast

import torch
from torch import Tensor

from cirkit.new.layers.input.exp_family.exp_family import ExpFamilyLayer
from cirkit.new.reparams import EFNormalReparam


class NormalLayer(ExpFamilyLayer):
    """The Normal distribution layer.

    This is fully factorized down to univariate Categorical distributions.
    """

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        reparam: EFNormalReparam,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (EFNormalReparam): The reparameterization for layer parameters. Must be \
                EFNormalReparam.
        """
        assert isinstance(reparam, EFNormalReparam), "Must use a EFNormalReparam for NormalLayer."
        self.suff_stats_shape = (2, num_input_units)  # 2 for mean and var.
        # Set self.suff_stats_shape before ExpFamilyLayer.__init__. Although dim=-1 is marked to
        # normalize there, the reparam is expected to be EFNormalReparam which ignores the dim.
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        # log_h is a fixed value, so calc in __init__ instead of on the fly. We don't convert to
        # Tensor now so that the default_dtype may still be changed later.
        self._log_h = -0.5 * self.num_input_units * math.log(2 * math.pi)

    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, Ki).

        Returns:
            Tensor: The sufficient statistics T, shape (H, *B, *S).
        """
        # TODO: torch __pow__ issue
        return torch.stack((x, cast(Tensor, x**2)), dim=-2)  # shape (H, *B, 2, Ki).

    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, Ki).

        Returns:
            Tensor: The natural parameters eta, shape (H, *B).
        """
        return x.new_full((), self._log_h).expand(x.shape[:-1])

    def log_partition(self, eta: Tensor, *, eta_normed: bool = False) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (H, Ko, *S).
            eta_normed (bool, optional): Ignored. This layer uses a special reparam for eta. \
                Defaults to False.

        Returns:
            Tensor: The log partition function A, shape (H, Ko).
        """
        eta1 = eta[..., 0, :]  # shape (H, Ko, Ki).
        eta2 = eta[..., 1, :]  # shape (H, Ko, Ki).
        # TODO: torch __pow__ issue
        # shape (H, Ko, Ki) -> (H, Ko).
        return torch.sum(
            -0.25 * cast(Tensor, eta1**2) / eta2 - 0.5 * torch.log(-2 * eta2), dim=-1
        )
