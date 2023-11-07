from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F

from cirkit.region_graph import RegionNode

from .exp_family import ExpFamilyLayer

# TODO: rework docstrings


class BinomialLayer(ExpFamilyLayer):
    """Implementation of Binomial distribution."""

    def __init__(self, nodes: List[RegionNode], num_channels: int, num_units: int, *, n: int):
        """Init class.

        Args:
            nodes (List[RegionNode]): Passed to super.
            num_channels (int): Number of dims.
            num_units (int): The number of units.
            n (int): n for binomial.
        """
        super().__init__(nodes, num_channels, num_units, num_stats=num_channels)
        self.n = n

    def reparam_function(self, params: Tensor) -> Tensor:
        """Do reparam.

        Args:
            params (Tensor): The params.

        Returns:
            Tensor: Reparams.
        """
        return torch.sigmoid(params * 0.1) * self.n

    def sufficient_statistics(self, x: Tensor) -> Tensor:
        """Get sufficient statistics.

        Args:
            x (Tensor): The input.

        Returns:
            Tensor: The stats.
        """
        assert len(x.shape) == 2 or len(x.shape) == 3, "Input must be 2 or 3 dimensional tensor."
        return x.unsqueeze(-1) if len(x.shape) == 2 else x

    # TODO: properly doc x, phi, theta...?
    def expectation_to_natural(self, phi: Tensor) -> Tensor:
        """Get expectation to natural.

        Args:
            phi (Tensor): The input.

        Returns:
            Tensor: The expectation.
        """
        theta = torch.clamp(phi / self.n, 1e-6, 1 - 1e-6)
        # TODO: is this a mypy bug?
        theta = torch.log(theta) - torch.log(1 - theta)  # type: ignore[misc]
        return theta

    def log_normalizer(self, theta: Tensor) -> Tensor:
        """Get normalizer.

        Args:
            theta (Tensor): The input.

        Returns:
            Tensor: The normalizer.
        """
        # TODO: issue with pylint on torch?
        return torch.sum(F.softplus(theta), dim=-1)  # pylint: disable=not-callable

    def log_h(self, x: Tensor) -> Tensor:
        """Get log h.

        Args:
            x (Tensor): the input.

        Returns:
            Tensor: The output.
        """
        if self.n == 1:
            return torch.zeros(()).to(x)

        log_h = (
            torch.lgamma(torch.tensor(self.n + 1).to(x))
            - torch.lgamma(x + 1)
            # TODO: is this a mypy bug?
            - torch.lgamma(self.n + 1 - x)  # type: ignore[misc]
        )
        if len(x.shape) == 3:
            log_h = log_h.sum(dim=-1)
        return log_h
