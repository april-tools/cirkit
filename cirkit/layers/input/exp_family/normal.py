import math
from typing import List, cast

import torch
from torch import Tensor

from cirkit.region_graph import RegionNode
from cirkit.reparams.leaf import ReparamEFNormal
from cirkit.utils.type_aliases import ReparamFactory

from .exp_family import ExpFamilyLayer

# TODO: rework docstrings


class NormalLayer(ExpFamilyLayer):
    """Implementation of Normal distribution."""

    def __init__(
        self,
        nodes: List[RegionNode],
        num_channels: int,
        num_units: int,
        *,
        reparam: ReparamFactory = ReparamEFNormal,
    ):
        """Init class.

        Args:
            nodes (List[RegionNode]): Passed to super.
            num_channels (int): Number of dims.
            num_units (int): Number of units.
            reparam (ReparamFactory): reparam.
        """
        super().__init__(
            nodes, num_channels, num_units, num_stats=2 * num_channels, reparam=reparam
        )

    def natural_params(self, theta: Tensor) -> Tensor:
        """Calculate natural parameters eta from parameters theta.

        Args:
            theta (Tensor): The parameters theta, shape (D, K, P, S).

        Returns:
            Tensor: The natural parameters eta, shape (D, K, P, S).
        """
        mu = theta[..., : self.num_channels]  # shape (D, K, P, C)
        # TODO: torch __pow__ issue
        var = theta[..., self.num_channels :] - cast(Tensor, mu**2)  # shape (D, K, P, C)
        eta = torch.stack(
            (mu, torch.tensor(-0.5).to(mu).expand_as(mu)), dim=-2
        )  # shape (D, K, P, 2, C)
        return (eta / var.unsqueeze(dim=-2)).flatten(start_dim=-2)  # shape (D, K, P, S=2*C)

    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (B, D, C).

        Returns:
            Tensor: The sufficient statistics T, shape (B, D, S).
        """
        # TODO: torch __pow__ issue
        return torch.cat((x, x**2), dim=-1)  # type: ignore[misc]  # shape (B, D, S=2*C)

    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (B, D, C).

        Returns:
            Tensor: The natural parameters eta, shape (B, D).
        """
        return (
            torch.tensor(-0.5 * self.num_channels * math.log(2 * math.pi))
            .to(x)
            .expand_as(x[..., 0])
        )

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
        return log_normalizer.sum(dim=-1)
