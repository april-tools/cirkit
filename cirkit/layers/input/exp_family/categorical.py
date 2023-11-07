from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F

from cirkit.region_graph import RegionNode
from cirkit.reparams.leaf import ReparamLogSoftmax
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
        reparam: ReparamFactory = ReparamLogSoftmax,
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

    def natural_params(self, theta: Tensor) -> Tensor:
        """Calculate natural parameters eta from parameters theta.

        Args:
            theta (Tensor): The parameters theta, shape (D, K, P, S).

        Returns:
            Tensor: The natural parameters eta, shape (D, K, P, S).
        """
        # TODO: not sure what will happen with C>1
        # TODO: x.unflatten is not typed
        theta = torch.unflatten(
            theta, dim=-1, sizes=(self.num_channels, self.num_categories)
        )  # shape (D, K, P, C, cat)
        theta = theta - theta.logsumexp(dim=-1, keepdim=True)
        return theta.flatten(start_dim=-2)  # shape (D, K, P, S=C*cat)

    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (B, D, C).

        Returns:
            Tensor: The sufficient statistics T, shape (B, D, S).
        """
        if x.is_floating_point():
            x = x.long()
        # TODO: pylint issue?
        # pylint: disable-next=not-callable
        suff_stats = F.one_hot(x, self.num_categories).float()  # shape (B, D, C, cat)
        return suff_stats.flatten(start_dim=-2)  # shape (B, D, S=C*cat)

    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (B, D, C).

        Returns:
            Tensor: The natural parameters eta, shape (B, D).
        """
        return torch.zeros(()).to(x).expand_as(x[..., 0])

    def log_partition(self, eta: Tensor) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (D, K, P, S).

        Returns:
            Tensor: The log partition function A, shape (D, K, P).
        """
        return torch.zeros(()).to(eta).expand_as(eta[..., 0])
