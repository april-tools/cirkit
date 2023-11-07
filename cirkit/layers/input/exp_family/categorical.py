from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F

from cirkit.region_graph import RegionNode
from cirkit.reparams.leaf import ReparamSoftmax
from cirkit.utils.type_aliases import ReparamFactory

from .exp_family import ExpFamilyLayer

# TODO: rework docstrings


class CategoricalLayer(ExpFamilyLayer):
    """Implementation of Categorical distribution."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        rg_nodes: List[RegionNode],
        num_channels: int,
        num_units: int,
        *,
        num_categories: int,
        reparam: ReparamFactory = ReparamSoftmax,
    ):
        """Init class.

        Args:
            rg_nodes (List[RegionNode]): Passed to super.
            num_channels (int): Number of dims.
            num_units (int): Number of input units,
            num_categories (int): k for category.
            reparam (ReparamFactory): reparam.
        """
        super().__init__(
            rg_nodes,
            num_channels,
            num_units,
            num_stats=num_channels * num_categories,
            reparam=reparam,
        )
        self.num_categories = num_categories

    def sufficient_statistics(self, x: Tensor) -> Tensor:
        """Get sufficient statistics.

        Args:
            x (Tensor): The input.

        Returns:
            Tensor: The stats.
        """
        # TODO: do we put this assert in super()?
        assert len(x.shape) == 2 or len(x.shape) == 3, "Input must be 2 or 3 dimensional tensor."

        if x.is_floating_point():
            x = x.long()
        # TODO: pylint issue?
        stats = F.one_hot(x, self.num_categories).float()  # pylint: disable=not-callable
        return (
            stats.reshape(-1, self.num_channels * self.num_categories)
            if len(x.shape) == 3
            else stats
        )

    def expectation_to_natural(self, phi: Tensor) -> Tensor:
        """Get expectation to natural.

        Args:
            phi (Tensor): The input.

        Returns:
            Tensor: The expectation.
        """
        # TODO: how to save the shape
        array_shape = self.params.shape[1:3]
        theta = torch.clamp(phi, 1e-12, 1)
        theta = theta.reshape(self.num_vars, *array_shape, self.num_channels, self.num_categories)
        theta /= theta.sum(dim=-1, keepdim=True)
        theta = theta.reshape(self.num_vars, *array_shape, self.num_channels * self.num_categories)
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
