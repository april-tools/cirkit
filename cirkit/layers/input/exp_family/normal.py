import math
from typing import List

import torch
from torch import Tensor

from cirkit.region_graph import RegionNode

from .exp_family import ExpFamilyLayer

# TODO: rework docstrings


class NormalLayer(ExpFamilyLayer):
    """Implementation of Normal distribution."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        nodes: List[RegionNode],
        num_channels: int,
        num_units: int,
        *,
        min_var: float = 0.0001,
        max_var: float = 10.0,
    ):
        """Init class.

        Args:
            nodes (List[RegionNode]): Passed to super.
            num_channels (int): Number of dims.
            num_units (int): Number of units.
            min_var (float, optional): Min var. Defaults to 0.0001.
            max_var (float, optional): Max var. Defaults to 10.0.
        """
        super().__init__(nodes, num_channels, num_units, num_stats=2 * num_channels)
        self.min_var = min_var
        self.max_var = max_var
        self._log_h = torch.tensor(-0.5 * math.log(2 * math.pi) * self.num_channels)

    def reparam_function(self, params: Tensor) -> Tensor:
        """Get reparamed params.

        Args:
            params (Tensor): Params.

        Returns:
            Tensor: Re-params.
        """
        mu = params[..., : self.num_channels]
        var = (
            torch.sigmoid(params[..., self.num_channels :]) * (self.max_var - self.min_var)
            + self.min_var
        )
        # TODO: is this a mypy bug?
        return torch.cat((mu, var + mu**2), dim=-1)  # type: ignore[misc]

    def sufficient_statistics(self, x: Tensor) -> Tensor:
        """Get sufficient statistics.

        Args:
            x (Tensor): The input.

        Returns:
            Tensor: The stats.
        """
        assert len(x.shape) == 2 or len(x.shape) == 3, "Input must be 2 or 3 dimensional tensor."
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        # TODO: is this a mypy bug?
        return torch.cat((x, x**2), dim=-1)  # type: ignore[misc]

    def expectation_to_natural(self, phi: Tensor) -> Tensor:
        """Get expectation.

        Args:
            phi (Tensor): The phi? I don't know.

        Returns:
            Tensor: The expectation.
        """
        # TODO: is this a mypy bug?
        var: Tensor = phi[..., : self.num_channels] ** 2
        var = phi[..., self.num_channels :] - var
        theta1 = phi[..., : self.num_channels] / var
        # TODO: another mypy bug? 2*var is ok, but -1/() is Any
        theta2: Tensor = -1 / (2 * var)
        return torch.cat((theta1, theta2), dim=-1)

    def log_normalizer(self, theta: Tensor) -> Tensor:
        """Get the norm for log.

        Args:
            theta (Tensor): The input.

        Returns:
            Tensor: The normalizer.
        """
        # TODO: is this a mypy bug?
        log_normalizer: Tensor = theta[..., : self.num_channels] ** 2 / (  # type: ignore[misc]
            -4 * theta[..., self.num_channels :]
        ) - 0.5 * torch.log(-2 * theta[..., self.num_channels :])
        log_normalizer = torch.sum(log_normalizer, dim=-1)
        return log_normalizer

    def log_h(self, x: Tensor) -> Tensor:
        """Get log h.

        Args:
            x (Tensor): The input.

        Returns:
            Tensor: The log_h.
        """
        return self._log_h.to(x)
