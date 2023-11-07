from abc import abstractmethod
from typing import List

import torch
from torch import Tensor, nn

from cirkit.layers.input import InputLayer
from cirkit.region_graph.rg_node import RegionNode

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

    def __init__(
        self,
        rg_nodes: List[RegionNode],
        num_channels: int,
        num_units: int,
        num_stats: int = 1,
    ):
        """Init class.

        :param rg_nodes: list of PC leaves (DistributionVector, see Graph.py)
        :param num_vars: number of random variables (int)
        :param num_channels: dimensionality of RVs (int)
        :param num_units: The number of units (int).
        :param num_stats: number of sufficient statistics of exponential family (int)
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

        params_shape = (self.num_vars, self.num_units, self.num_replicas, self.num_stats)
        self.params = nn.Parameter(torch.empty(params_shape))

        # if em is switched off, we re-parametrize the expectation parameters
        # self.reparam holds the function object for this task
        self.reparam = self.reparam_function
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters to default initialization: N(0, 1)."""
        nn.init.normal_(self.params)

    def integrate(self) -> Tensor:
        """Return the integation, which is a zero tensor for this layer (in log-space).

        Returns:
            Tensor: A zero tensor of shape (1, num_vars, num_units, num_replicas).
        """
        return torch.zeros(
            size=(1, self.num_vars, self.num_units, self.num_replicas),
            requires_grad=False,
            device=self.params.device,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute the factorized leaf densities. We are doing the computation \
            in the log-domain, so this is actually \
            computing sums over densities.

        We first pass the data x into self.ef_array, which computes a tensor of shape
            (batch_size, num_vars, num_dist, num_replica). This is best interpreted \
            as vectors of length num_dist, for each \
            sample in the batch and each RV. Since some leaves have overlapping \
            scope, we need to compute "enough" leaves, \
            hence the num_replica dimension. The assignment of these log-densities \
            to leaves is represented with \
            self.scope_tensor.
        In the end, the factorization (sum in log-domain) is realized with a single einsum.

        :param x: input data (Tensor).
                  If self.num_channels == 1, this can be either of shape \
                    (batch_size, self.num_vars, 1) or
                  (batch_size, self.num_vars).
                  If self.num_channels > 1, this must be of shape \
                    (batch_size, self.num_vars, self.num_channels).
        :return: log-density vectors of leaves
                 Will be of shape (batch_size, num_dist, len(self.rg_nodes))
                 Note: num_dist is K in the paper, len(self.rg_nodes) is the number of PC leaves
        """
        # Re-parametrize first
        # TODO: no_grad? the deleted self.reparam==None branch have no_grad
        phi = self.reparam(self.params)

        # Convert to natural parameters
        # theta: (num_vars, num_units, num_replicas, num_stats)
        theta = self.expectation_to_natural(phi)

        # Compute sufficient statistics
        # suff_stats: (batch_size, num_vars, num_stats)
        suff_stats = self.sufficient_statistics(x)

        # Compute the log normalizer
        # log_normalizer: (num_vars, num_units, num_replicas)
        log_normalizer = self.log_normalizer(theta)

        # Compute the log h(x) values
        # log_h: scalar or (batch_size, num_vars, 1, 1)
        log_h = self.log_h(x)
        if len(log_h.shape) > 0:
            log_h = log_h.unsqueeze(dim=2).unsqueeze(dim=3)

        # Compute the product of natural parameters and sufficient statistics.
        # Moreover, sum over channel dimensions, which translates to naive factorization
        theta_suff_stats = torch.einsum("dipj,bdj->bdip", theta, suff_stats)

        # Finally compute the log-likelihoods
        # log_probs: (batch_size, num_vars, num_units, num_replicas)
        log_probs = log_h + theta_suff_stats - log_normalizer
        return log_probs

    @abstractmethod
    def sufficient_statistics(self, x: Tensor) -> Tensor:
        """Get sufficient statistics for the implemented exponential \
            family (called T(x) in the paper).

        :param x: observed data (Tensor).
                  If self.num_channels == 1, this can be either of shape \
                    (batch_size, self.num_vars, 1) or
                  (batch_size, self.num_vars).
                  If self.num_channels > 1, this must be of shape \
                    (batch_size, self.num_vars, self.num_channels).
        :return: sufficient statistics of the implemented exponential family (Tensor).
                 Must be of shape (batch_size, self.num_vars, self.num_stats)
        """

    @abstractmethod
    def log_normalizer(self, theta: Tensor) -> Tensor:
        """Log-normalizer of the implemented exponential family (called A(theta) in the paper).

        :param theta: natural parameters (Tensor). Must be of shape \
            (self.num_vars, *self.array_shape, self.num_stats).
        :return: log-normalizer (Tensor). Must be of shape (self.num_vars, *self.array_shape).
        """

    @abstractmethod
    def log_h(self, x: Tensor) -> Tensor:
        """Get the log of the base measure (called h(x) in the paper).

        :param x: observed data (Tensor).
                  If self.num_channels == 1, this can be either of shape \
                    (batch_size, self.num_vars, 1) or
                  (batch_size, self.num_vars).
                  If self.num_channels > 1, this must be of shape \
                    (batch_size, self.num_vars, self.num_channels).
        :return: log(h) of the implemented exponential family (Tensor).
                 Can either be a scalar or must be of shape (batch_size, self.num_vars)
        """

    @abstractmethod
    def expectation_to_natural(self, phi: Tensor) -> Tensor:
        """Conversion from expectations parameters phi to natural parameters \
            theta, for the implemented exponential family.

        :param phi: expectation parameters (Tensor). Must be of shape \
            (self.num_vars, *self.array_shape, self.num_stats).
        :return: natural parameters theta (Tensor). Same shape as phi.
        """

    @abstractmethod
    def reparam_function(self, params: Tensor) -> Tensor:
        """Re-parameterize parameters, in order that they stay in their constrained domain.

        When we are not using the EM, we need to transform unconstrained \
            (real-valued) parameters to the constrained set \
            of the expectation parameter. This function should return such a \
            function (i.e. the return value should not be \
            a projection, but a function which does the projection).

        :param params: I don't know
        :return: function object f which takes as input unconstrained parameters (Tensor) \
            and returns re-parametrized parameters.
        """

    # TODO: see 241d46a43f59c1df23b5136a45b5f18b9f116671 for backtrack
