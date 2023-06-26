import math
from typing import Any, List

import torch
from torch import Tensor

from cirkit.region_graph import RegionNode

from .exp_family import ExpFamilyLayer

# TODO: rework docstrings


# TODO: better way to permute?
def _shift_last_axis_to(x: Tensor, i: int) -> Tensor:
    """Take the last axis of tensor x and inserts it at position i."""
    num_axes = len(x.shape)
    return x.permute(tuple(range(i)) + (num_axes - 1,) + tuple(range(i, num_axes - 1)))


class NormalLayer(ExpFamilyLayer):
    """Implementation of Normal distribution."""

    def __init__(
        self,
        nodes: List[RegionNode],
        num_var: int,
        num_dims: int,
        num_units: int,
        *,
        min_var: float = 0.0001,
        max_var: float = 10.0,
    ):
        """Init class.

        Args:
            nodes (List[RegionNode]): Passed to super.
            num_var (int): Number of vars.
            num_dims (int): Number of dims.
            num_units (int): Number of units.
            min_var (float, optional): Min var. Defaults to 0.0001.
            max_var (float, optional): Max var. Defaults to 10.0.
        """
        super().__init__(nodes, num_var, num_dims, num_units, num_stats=2 * num_dims)
        self.min_var = min_var
        self.max_var = max_var
        self._log_h = torch.tensor(-0.5 * math.log(2 * math.pi) * self.num_dims)

    def reparam_function(self, params: Tensor) -> Tensor:
        """Get reparamed params.

        Args:
            params (Tensor): Params.

        Returns:
            Tensor: Re-params.
        """
        mu = params[..., : self.num_dims]
        var = (
            torch.sigmoid(params[..., self.num_dims :]) * (self.max_var - self.min_var)
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
        var: Tensor = phi[..., : self.num_dims] ** 2
        var = phi[..., self.num_dims :] - var
        theta1 = phi[..., : self.num_dims] / var
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
        log_normalizer: Tensor = theta[..., : self.num_dims] ** 2 / (  # type: ignore[misc]
            -4 * theta[..., self.num_dims :]
        ) - 0.5 * torch.log(-2 * theta[..., self.num_dims :])
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

    # TODO: for now inherit parent docstring
    def _sample(  # type: ignore[misc]
        self, num_samples: int, params: Tensor, std_correction: float = 1.0, **_: Any
    ) -> Tensor:
        # TODO: no_grad on decorator?
        with torch.no_grad():
            mu = params[..., : self.num_dims]
            # TODO: is this a mypy bug?
            std = torch.sqrt(params[..., self.num_dims :] - mu**2)  # type: ignore[misc]
            # TODO: same dtype device idiom?
            samples = mu.unsqueeze(0) + std_correction * std.unsqueeze(0) * torch.randn(
                num_samples, *mu.shape, dtype=mu.dtype, device=mu.device
            )
            return _shift_last_axis_to(samples, 2)

    # TODO: do we allow explicit any?
    def _argmax(self, params: Tensor, **_: Any) -> Tensor:  # type: ignore[misc]
        with torch.no_grad():
            mu = params[..., : self.num_dims]
            return _shift_last_axis_to(mu, 1)
