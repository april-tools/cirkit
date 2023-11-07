from abc import abstractmethod
from typing import List

import torch
from torch import Tensor, nn

from cirkit.layers.input import InputLayer
from cirkit.region_graph.rg_node import RegionNode
from cirkit.utils.type_aliases import ReparamFactory

# TODO: find a good way to doc tensor shape
# TODO: rework docstrings


# TODO: but we don't have a non-factorized one
class ExpFamilyLayer(InputLayer):
    """Computes all EiNet leaves in parallel, where each leaf is a vector of \
        factorized distributions, where factors are from exponential families.

    In FactorizedLeafLayer, we generate an ExponentialFamilyArray with \
        array_shape = (num_dist, num_replica), where
        num_dist is the vector length of the vectorized distributions \
            (K in the paper), and
        num_replica is picked large enough such that "we compute enough \
            leaf densities". At the moment we rely that
        the PC structure (see Class Graph) provides the necessary information \
            to determine num_replica.
        In the future, it would be convenient to have an automatic allocation \
            of leaves to replica, without requiring
        the user to specify this.
    The generate ExponentialFamilyArray has shape (batch_size, num_vars, \
        num_dist, num_replica). This array of densities
        will contain all densities over single RVs, which are then multiplied \
        (actually summed, due to log-domain
        computation) together in forward(...).
    """

    # TODO: keep the natural_params and calc the inverse mapping for params
    @abstractmethod
    def natural_params(self, theta: Tensor) -> Tensor:
        """Calculate natural parameters eta from parameters theta.

        Args:
            theta (Tensor): The parameters theta, shape (D, K, P, S).

        Returns:
            Tensor: The natural parameters eta, shape (D, K, P, S).
        """

    @abstractmethod
    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (B, D, C).

        Returns:
            Tensor: The sufficient statistics T, shape (B, D, S).
        """

    @abstractmethod
    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (B, D, C).

        Returns:
            Tensor: The natural parameters eta, shape (B, D).
        """

    @abstractmethod
    def log_partition(self, eta: Tensor) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (D, K, P, S).

        Returns:
            Tensor: The log partition function A, shape (D, K, P).
        """

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (B, D, C).

        Returns:
            Tensor: The output of this layer, shape (B, D, K, P).
        """
        # TODO: this does not work for more than 1 batch dims
        if x.ndim == 2:
            x = x.unsqueeze(dim=-1)

        eta = self.natural_params(self.params())  # shape (D, K, P, S)
        suff_stats = self.sufficient_stats(x)  # shape (B, D, S)
        log_h = self.log_base_measure(x)  # shape (B, D)
        log_part = self.log_partition(eta)  # shape (D, K, P)
        return (
            torch.einsum("dkps,bds->bdkp", eta, suff_stats)  # shape (B, D, K, P)
            + log_h.unsqueeze(dim=-1).unsqueeze(dim=-1)  # shape (B, D, 1, 1)
            - log_part.unsqueeze(dim=0)  # shape (1, D, K, P)
        )  # shape (B, D, K, P)

    ###################################################################
    ###################################################################
    ###################################################################
    ###################################################################

    def __init__(  # pylint: disable=too-many-arguments
        self,
        rg_nodes: List[RegionNode],
        num_channels: int,
        num_units: int,
        num_stats: int = 1,
        *,
        reparam: ReparamFactory,
    ):
        """Init class.

        :param rg_nodes: list of PC leaves (DistributionVector, see Graph.py)
        :param num_vars: number of random variables (int)
        :param num_channels: dimensionality of RVs (int)
        :param num_units: The number of units (int).
        :param num_stats: number of sufficient statistics of exponential family (int)
        :param reparam: reparams (ReparamFactory)
        """
        super().__init__(rg_nodes)
        self.num_channels = num_channels
        self.num_units = num_units
        self.num_stats = num_stats

        replica_indices = set(n.get_replica_idx() for n in rg_nodes)
        self.num_replicas = len(replica_indices)
        assert replica_indices == set(
            range(self.num_replicas)
        ), "Replica indices should be consecutive, starting with 0."

        self.params = reparam(
            (self.num_vars, self.num_units, self.num_replicas, self.num_stats), dim=-1
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset parameters to default: N(0, 1)."""
        for param in self.parameters():
            nn.init.normal_(param, 0, 1)

    def integrate(self) -> Tensor:
        """Return the integation, which is a zero tensor for this layer (in log-space).

        Returns:
            Tensor: A zero tensor of shape (1, num_vars, num_units, num_replicas).
        """
        return torch.zeros(
            size=(1, self.num_vars, self.num_units, self.num_replicas),
            requires_grad=False,
            device=self.params().device,  # TODO: this is not good
        )

    # TODO: see 241d46a43f59c1df23b5136a45b5f18b9f116671 for backtrack
