import math
from typing import Any, Sequence

import torch
from torch import Tensor

from .exp_family_array import ExponentialFamilyArray


# TODO: better way to permute?
def _shift_last_axis_to(x: Tensor, i: int) -> Tensor:
    """Take the last axis of tensor x and inserts it at position i."""
    num_axes = len(x.shape)
    return x.permute(tuple(range(i)) + (num_axes - 1,) + tuple(range(i, num_axes - 1)))


class NormalArray(ExponentialFamilyArray):
    """Implementation of Normal distribution."""

    def __init__(  # TODO: pylint: disable=too-many-arguments
        self,
        num_var: int,
        num_dims: int,
        array_shape: Sequence[int],
        min_var: float = 0.0001,
        max_var: float = 10.0,
        use_em: bool = True,
    ):
        """Init class.

        Args:
            num_var (int): Number of vars.
            num_dims (int): Number of dims.
            array_shape (Sequence[int]): Shape of array.
            min_var (float, optional): Min var. Defaults to 0.0001.
            max_var (float, optional): Max var. Defaults to 10..
            use_em (bool, optional): Whether to use EM. Defaults to True.
        """
        super().__init__(num_var, num_dims, array_shape, num_stats=2 * num_dims, use_em=use_em)
        self.min_var = min_var
        self.max_var = max_var
        self._log_h = torch.tensor(-0.5 * math.log(2 * math.pi) * self.num_dims)

    def default_initializer(self) -> Tensor:
        """Init by default.

        Returns:
            Tensor: The default init.
        """
        phi = torch.empty(self.num_var, *self.array_shape, 2 * self.num_dims)
        phi[..., 0 : self.num_dims] = torch.randn(self.num_var, *self.array_shape, self.num_dims)
        # TODO: is this a mypy bug? phi[..., 0 : self.num_dims] ** 2 is Any
        # but phi[..., 0 : self.num_dims] is Tensor
        phi[..., self.num_dims :] = 1 + phi[..., 0 : self.num_dims] ** 2  # type: ignore[misc]
        return phi

    def project_params(self, params: Tensor) -> Tensor:
        """Project params.

        Args:
            params (Tensor): Params to project.

        Returns:
            Tensor: Projected params.
        """
        params_project = params.clone()
        # TODO: redundant annotation is this a mypy bug? same as above
        mu2: Tensor = params_project[..., 0 : self.num_dims] ** 2
        params_project[..., self.num_dims :] = torch.clamp(
            params_project[..., self.num_dims :], mu2 + self.min_var, mu2 + self.max_var
        )
        # TODO: which one is better?
        # params_project[..., self.num_dims :] -= mu2
        # params_project[..., self.num_dims :] = torch.clamp(
        #     params_project[..., self.num_dims :], self.min_var, self.max_var
        # )
        # params_project[..., self.num_dims :] += mu2
        return params_project

    def reparam_function(self, params: Tensor) -> Tensor:
        """Get reparamed params.

        Args:
            params (Tensor): Params.

        Returns:
            Tensor: Re-params.
        """
        mu = params[..., 0 : self.num_dims]
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
        var: Tensor = phi[..., 0 : self.num_dims] ** 2
        var = phi[..., self.num_dims :] - var
        theta1 = phi[..., 0 : self.num_dims] / var
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
        log_normalizer: Tensor = theta[..., 0 : self.num_dims] ** 2 / (  # type: ignore[misc]
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
            mu = params[..., 0 : self.num_dims]
            # TODO: is this a mypy bug?
            std = torch.sqrt(params[..., self.num_dims :] - mu**2)  # type: ignore[misc]
            shape = (num_samples,) + mu.shape
            # TODO: same dtype device idiom?
            samples = mu.unsqueeze(0) + std_correction * std.unsqueeze(0) * torch.randn(
                shape, dtype=mu.dtype, device=mu.device
            )
            return _shift_last_axis_to(samples, 2)

    # TODO: do we allow explicit any?
    def _argmax(self, params: Tensor, **_: Any) -> Tensor:  # type: ignore[misc]
        with torch.no_grad():
            mu = params[..., 0 : self.num_dims]
            return _shift_last_axis_to(mu, 1)
