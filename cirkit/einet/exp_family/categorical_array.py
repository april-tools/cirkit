from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .exp_family_array import ExponentialFamilyArray
from .normal_array import _shift_last_axis_to


@torch.no_grad()
def _one_hot(x: Tensor, k: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """One hot encoding."""
    ind = torch.zeros(x.shape + (k,), dtype=dtype, device=x.device)
    ind.scatter_(dim=-1, index=x.unsqueeze(-1), value=1)
    return ind


class CategoricalArray(ExponentialFamilyArray):
    """Implementation of Categorical distribution."""

    def __init__(  # pylint: disable=too-many-arguments
        self, num_var: int, num_dims: int, array_shape: Sequence[int], k: int, use_em: bool = True
    ):
        """Init class.

        Args:
            num_var (int): Number of vars.
            num_dims (int): Number of dims.
            array_shape (Sequence[int]): Shape of array.
            k (int): k for category.
            use_em (bool, optional): Whether to use EM. Defaults to True.
        """
        super().__init__(num_var, num_dims, array_shape, num_stats=num_dims * k, use_em=use_em)
        self.k = k

    def default_initializer(self) -> Tensor:
        """Init by default.

        Returns:
            Tensor: The init.
        """
        # TODO: this shape is highly repeated. save it?
        return 0.01 + 0.98 * torch.rand(self.num_var, *self.array_shape, self.num_dims * self.k)

    def project_params(self, params: Tensor) -> Tensor:
        """Project params.

        Note that this is not actually l2-projection. For simplicity, we simply renormalize.

        Args:
            params (Tensor): The params.

        Returns:
            Tensor: Projected params.
        """
        params = params.reshape(self.num_var, *self.array_shape, self.num_dims, self.k)
        params = torch.clamp(params, min=1e-12)
        params = params / torch.sum(params, dim=-1, keepdim=True)
        return params.reshape(self.num_var, *self.array_shape, self.num_dims * self.k)

    def reparam_function(self, params: Tensor) -> Tensor:
        """Do reparam.

        Args:
            params (Tensor): The params.

        Returns:
            Tensor: Reparams.
        """
        return F.softmax(params, dim=-1)

    def sufficient_statistics(self, x: Tensor) -> Tensor:
        """Get sufficient statistics.

        Args:
            x (Tensor): The input.

        Returns:
            Tensor: The stats.
        """
        # TODO: do we put this assert in super()?
        assert len(x.shape) == 2 or len(x.shape) == 3, "Input must be 2 or 3 dimensional tensor."

        stats = _one_hot(x.long(), self.k)
        return stats.reshape(-1, self.num_dims * self.k) if len(x.shape) == 3 else stats

    def expectation_to_natural(self, phi: Tensor) -> Tensor:
        """Get expectation to natural.

        Args:
            phi (Tensor): The input.

        Returns:
            Tensor: The expectation.
        """
        theta = torch.clamp(phi, 1e-12, 1)
        theta = theta.reshape(self.num_var, *self.array_shape, self.num_dims, self.k)
        theta /= theta.sum(dim=-1, keepdim=True)
        theta = theta.reshape(self.num_var, *self.array_shape, self.num_dims * self.k)
        theta = torch.log(theta)
        return theta

    def log_normalizer(self, theta: Tensor) -> Tensor:
        """Get normalizer.

        Args:
            theta (Tensor): The input.

        Returns:
            Tensor: The normalizer.
        """
        return torch.zeros(()).to(theta)

    def log_h(self, x: Tensor) -> Tensor:
        """Get log h.

        Args:
            x (Tensor): the input.

        Returns:
            Tensor: The output.
        """
        return torch.zeros(()).to(x)

    def _sample(  # type: ignore[misc]
        self, num_samples: int, params: Tensor, dtype: torch.dtype = torch.float32, **_: Any
    ) -> Tensor:
        with torch.no_grad():
            dist = params.reshape(self.num_var, *self.array_shape, self.num_dims, self.k)
            cum_sum = torch.cumsum(dist[..., 0:-1], dim=-1)  # TODO: why slice to -1?
            rand = torch.rand((num_samples,) + cum_sum.shape[0:-1] + (1,), device=cum_sum.device)
            samples = torch.sum(rand > cum_sum, dim=-1).to(dtype)
            return _shift_last_axis_to(samples, 2)

    # TODO: why pass in dtype instead of cast outside?
    def _argmax(  # type: ignore[misc]
        self, params: Tensor, dtype: torch.dtype = torch.float32, **_: Any
    ) -> Tensor:
        with torch.no_grad():
            dist = params.reshape(self.num_var, *self.array_shape, self.num_dims, self.k)
            mode = torch.argmax(dist, dim=-1).to(dtype)
            return _shift_last_axis_to(mode, 1)
