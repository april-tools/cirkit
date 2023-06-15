from abc import abstractmethod
from typing import Any, List, Literal, Optional, Sequence

import torch
from torch import Tensor, nn

from cirkit.region_graph.rg_node import RegionNode

from ..layer import Layer

# TODO: find a good way to doc tensor shape
# TODO: rework docstrings


# TODO: but we don't have a non-factorized one
class ExpFamilyInputLayer(Layer):  # pylint: disable=too-many-instance-attributes
    """Computes all EiNet leaves in parallel, where each leaf is a vector of \
        factorized distributions, where factors are from exponential families.

    In FactorizedLeafLayer, we generate an ExponentialFamilyArray with \
        array_shape = (num_dist, num_replica), where
        num_dist is the vector length of the vectorized distributions \
            (K in the paper), and
        num_replica is picked large enough such that "we compute enough \
            leaf densities". At the moment we rely that
        the PC structure (see Class Graph) provides the necessary information \
            to determine num_replica. In
        particular, we require that each leaf of the graph has the field \
            einet_address.replica_idx defined;
        num_replica is simply the max over all einet_address.replica_idx.
        In the future, it would convenient to have an automatic allocation \
            of leaves to replica, without requiring
        the user to specify this.
    The generate ExponentialFamilyArray has shape (batch_size, num_var, \
        num_dist, num_replica). This array of densities
        will contain all densities over single RVs, which are then multiplied \
        (actually summed, due to log-domain
        computation) together in forward(...).
    """

    scope_tensor: Tensor  # to be registered as buffer

    def __init__(
        self,
        nodes: List[RegionNode],
        num_var: int,
        num_dims: int,
        num_stats: int,
    ):
        """Init class.

        :param nodes: list of PC leaves (DistributionVector, see Graph.py)
        :param num_var: number of random variables (int)
        :param num_dims: dimensionality of RVs (int)
        :param num_stats: number of sufficient statistics of exponential family (int)
        """
        super().__init__()

        self.nodes = nodes
        self.num_var = num_var
        self.num_dims = num_dims

        num_dists = set(n.k for n in self.nodes)
        assert len(num_dists) == 1, "All leaves must have the same number of distributions."
        num_dist = num_dists.pop()

        replica_indices = set(n.einet_address.replica_idx for n in self.nodes)
        num_replica = len(replica_indices)
        assert replica_indices == set(
            range(num_replica)
        ), "Replica indices should be consecutive, starting with 0."

        # self.scope_tensor indicates which densities in self.ef_array belongs to which leaf.
        # TODO: it might be smart to have a sparse implementation --
        # I have experimented a bit with this, but it is not always faster.
        self.register_buffer("scope_tensor", torch.zeros(num_var, num_replica, len(self.nodes)))
        for i, node in enumerate(self.nodes):
            self.scope_tensor[
                list(node.scope), node.einet_address.replica_idx, i  # type: ignore[misc]
            ] = 1
            node.einet_address.layer = self
            node.einet_address.idx = i

        self.num_stats = num_stats
        self.params_shape = (num_var, num_dist, num_replica, num_stats)

        self.params = nn.Parameter(torch.empty(self.params_shape))

        # TODO: is this a good init? (originally None)
        self.ll: Tensor = torch.Tensor()
        self.suff_stats: Tensor = torch.Tensor()

        self.marginalization_idx: Optional[Tensor] = None  # TODO: should this be Tensor?
        self.marginalization_mask: Optional[Tensor] = None

        # TODO: types of all `None`s?
        # TODO: why allow None but not directly init?
        self._p_acc: Optional[Tensor] = None
        self._stats_acc: Optional[Tensor] = None

        # if em is switched off, we re-parametrize the expectation parameters
        # self.reparam holds the function object for this task
        self.reparam = self.reparam_function

        self.reset_parameters()

    @property
    def num_params(self) -> int:
        """Get number of params.

        Returns:
            int: The number of params.
        """
        return self.params.numel()

    def reset_parameters(self) -> None:
        """Reset parameters to default initialization: N(0, 1)."""
        nn.init.normal_(self.params)

    def forward(self, x: Optional[Tensor] = None) -> None:
        """Compute the factorized leaf densities. We are doing the computation \
            in the log-domain, so this is actually \
            computing sums over densities.

        We first pass the data x into self.ef_array, which computes a tensor of shape
            (batch_size, num_var, num_dist, num_replica). This is best interpreted \
            as vectors of length num_dist, for each \
            sample in the batch and each RV. Since some leaves have overlapping \
            scope, we need to compute "enough" leaves, \
            hence the num_replica dimension. The assignment of these log-densities \
            to leaves is represented with \
            self.scope_tensor.
        In the end, the factorization (sum in log-domain) is realized with a single einsum.

        :param x: input data (Tensor).
                  If self.num_dims == 1, this can be either of shape \
                    (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape \
                    (batch_size, self.num_var, self.num_dims).
        no return: log-density vectors of leaves
                 Will be of shape (batch_size, num_dist, len(self.nodes))
                 Note: num_dist is K in the paper, len(self.nodes) is the number of PC leaves
        """
        assert x is not None  # TODO: how we guarantee this?

        # TODO: no_grad? the deleted self.reparam==None branch have no_grad
        phi = self.reparam(self.params)

        # assert not torch.isnan(self.params).any()
        # assert not torch.isnan(phi).any()

        theta = self.expectation_to_natural(phi)

        # assert not torch.isnan(theta).any()
        # assert not torch.isinf(theta).any()

        # suff_stats: (batch_size, self.num_var, self.num_stats)
        self.suff_stats = self.sufficient_statistics(x)

        # assert not torch.isnan(self.suff_stats).any()
        # assert not torch.isinf(self.suff_stats).any()

        # log_normalizer: (self.num_var, *self.array_shape)
        log_normalizer = self.log_normalizer(theta)

        # log_h: scalar, or (batch_size, self.num_var)
        log_h = self.log_h(x)
        if len(log_h.shape) > 0:
            # reshape for broadcasting
            # TODO: this line is definitely not written in a good way
            log_h = log_h.reshape(log_h.shape[:2] + (1, 1))

        # compute the exponential family tensor
        # (batch_size, self.num_var, *self.array_shape)

        # antonio_mari -> edit: (theta.unsqueeze(0) * self.suff_stats).sum(-1) is inefficient
        # example: for MNIST with PD structure, batch_size=100 and num_sums=128,
        # categorical distr. it's computed
        # a tensor (100, 784, 128, 1, 256) -> over 10GB
        # given by the tensor broadcasting (1, 784, 128, 1, 256) * (100, 784, 1, 1, 256).
        # I try with an einsum operation (x, o, d, s), (b, x, s) -> b, x, o, d.
        # That should have the same result

        crucial_quantity_einsum = torch.einsum("xods,bxs->bxod", theta, self.suff_stats)

        # assert not torch.isnan(crucial_quantity_einsum).any()

        # reshape for broadcasting
        # shape = self.suff_stats.shape
        # shape = shape[0:2] + (1,) * len(self.array_shape) + (shape[2],)
        # self.suff_stats = self.suff_stats.reshape(shape)
        # crucial_quantity_orig = (theta.unsqueeze(0) * self.suff_stats).sum(-1)
        # assert torch.all(torch.eq(crucial_quantity_einsum, crucial_quantity_orig))
        # TODO: check also for other cases, for now I checked and it's correct

        # TODO: does ll have grad now?
        self.ll = log_h + crucial_quantity_einsum - log_normalizer

        # Marginalization in PCs works by simply setting leaves corresponding to
        # marginalized variables to 1 (0 in
        # (log-domain). We achieve this by a simple multiplicative 0-1 mask, generated here.
        # TODO: the marginalization mask doesn't need to be computed every time;
        # only when marginalization_idx changes.
        if self.marginalization_idx is not None:
            with torch.no_grad():
                # TODO: is this better? torch.ones(self.num_var).to(self.ll)
                self.marginalization_mask = torch.ones(
                    self.num_var, dtype=self.ll.dtype, device=self.ll.device
                )
                self.marginalization_mask[self.marginalization_idx] = 0
                # TODO: find another way to reshape
                shape = (1, self.num_var) + (1, 1)
                self.marginalization_mask = self.marginalization_mask.reshape(shape)
            output = self.ll * self.marginalization_mask
        else:
            self.marginalization_mask = None
            output = self.ll

        # assert not torch.isnan(output).any()
        # assert not torch.isinf(output).any()

        self.prob = torch.einsum("bxir,xro->bio", output, self.scope_tensor)

        # assert not torch.isnan(self.prob).any()
        # assert not torch.isinf(self.prob).any()

    # TODO: how to fix?
    # pylint: disable-next=arguments-differ
    def backtrack(  # type: ignore[misc]
        self,
        dist_idx: Sequence[Sequence[int]],  # TODO: can be iterable?
        node_idx: Sequence[Sequence[int]],
        *_: Any,
        mode: Literal["sample", "argmax"] = "sample",
        **kwargs: Any,
    ) -> Tensor:
        """Backtrackng mechanism for EiNets.

        :param dist_idx: list of N indices into the distribution vectors, which shall be sampled.
        :param node_idx: list of N indices into the leaves, which shall be sampled.
        :param mode: 'sample' or 'argmax'; for sampling or MPE approximation, respectively.
        :param _: ignored
        :param kwargs: keyword arguments
        :return: samples (Tensor). Of shape (N, self.num_var, self.num_dims).
        """
        assert len(dist_idx) == len(node_idx), "Invalid input."

        with torch.no_grad():
            big_n = len(dist_idx)  # TODO: a better name
            ef_values = (
                self.sample(big_n, **kwargs)  # type: ignore[misc]
                if mode == "sample"
                else self.argmax(**kwargs)  # type: ignore[misc]
            )

            values = torch.zeros(
                big_n, self.num_var, self.num_dims, device=ef_values.device, dtype=ef_values.dtype
            )

            # TODO: use enumerate?
            for n in range(big_n):
                cur_value = torch.zeros(
                    self.num_var, self.num_dims, device=ef_values.device, dtype=ef_values.dtype
                )
                assert len(dist_idx[n]) == len(node_idx[n]), "Invalid input."
                for c, k in enumerate(node_idx[n]):
                    scope = list(self.nodes[k].scope)
                    rep = self.nodes[k].einet_address.replica_idx
                    cur_value[scope, :] = (
                        ef_values[n, scope, :, dist_idx[n][c], rep]
                        if mode == "sample"
                        else ef_values[scope, :, dist_idx[n][c], rep]
                    )
                values[n, :, :] = cur_value  # TODO: directly slice this

            return values

    # TODO: why we need this for public attr?
    def set_marginalization_idx(self, idx: Tensor) -> None:
        """Set indicices of marginalized variables.

        Args:
            idx (Tensor): The indices.
        """
        self.marginalization_idx = idx

    # TODO: why optional?
    def get_marginalization_idx(self) -> Optional[Tensor]:
        """Get indicices of marginalized variables.

        Returns:
            Tensor: The indices.
        """
        return self.marginalization_idx

    def sample(self, num_samples: int = 1, **kwargs: Any) -> Tensor:  # type: ignore[misc]
        """Sample the dist.

        Args:
            num_samples (int, optional): Number of samples. Defaults to 1.
            kwargs (Any, optional): Any kwargs.

        Returns:
            Tensor: The sample.
        """
        # TODO: maybe the function should be no_grad?
        with torch.no_grad():
            params = self.reparam(self.params)
        return self._sample(num_samples, params, **kwargs)  # type: ignore[misc]

    def argmax(self, **kwargs: Any) -> Tensor:  # type: ignore[misc]
        """Get the argmax.

        Args:
            kwargs (Any, optional): Any kwargs.

        Returns:
            Tensor: The argmax.
        """
        with torch.no_grad():
            params = self.reparam(self.params)
        return self._argmax(params, **kwargs)  # type: ignore[misc]

    @abstractmethod
    def sufficient_statistics(self, x: Tensor) -> Tensor:
        """Get sufficient statistics for the implemented exponential \
            family (called T(x) in the paper).

        :param x: observed data (Tensor).
                  If self.num_dims == 1, this can be either of shape \
                    (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape \
                    (batch_size, self.num_var, self.num_dims).
        :return: sufficient statistics of the implemented exponential family (Tensor).
                 Must be of shape (batch_size, self.num_var, self.num_stats)
        """

    @abstractmethod
    def log_normalizer(self, theta: Tensor) -> Tensor:
        """Log-normalizer of the implemented exponential family (called A(theta) in the paper).

        :param theta: natural parameters (Tensor). Must be of shape \
            (self.num_var, *self.array_shape, self.num_stats).
        :return: log-normalizer (Tensor). Must be of shape (self.num_var, *self.array_shape).
        """

    @abstractmethod
    def log_h(self, x: Tensor) -> Tensor:
        """Get the log of the base measure (called h(x) in the paper).

        :param x: observed data (Tensor).
                  If self.num_dims == 1, this can be either of shape \
                    (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape \
                    (batch_size, self.num_var, self.num_dims).
        :return: log(h) of the implemented exponential family (Tensor).
                 Can either be a scalar or must be of shape (batch_size, self.num_var)
        """

    @abstractmethod
    def expectation_to_natural(self, phi: Tensor) -> Tensor:
        """Conversion from expectations parameters phi to natural parameters \
            theta, for the implemented exponential family.

        :param phi: expectation parameters (Tensor). Must be of shape \
            (self.num_var, *self.array_shape, self.num_stats).
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

    @abstractmethod
    def _sample(  # type: ignore[misc]
        self, num_samples: int, params: Tensor, **kwargs: Any
    ) -> Tensor:
        """Is helper function for sampling the exponential family.

        :param num_samples: number of samples to be produced
        :param params: expectation parameters (phi) of the exponential family, of shape
                       (self.num_var, *self.array_shape, self.num_stats)
        :param kwargs: keyword arguments
               Depending on the implementation, kwargs can also contain further arguments.
        :return: i.i.d. samples of the exponential family (Tensor).
                 Should be of shape (num_samples, self.num_var, self.num_dims, *self.array_shape)
        """

    @abstractmethod
    def _argmax(self, params: Tensor, **kwargs: Any) -> Tensor:  # type: ignore[misc]
        """Is helper function for getting the argmax of the exponential family.

        :param params: expectation parameters (phi) of the exponential family, of shape
                       (self.num_var, *self.array_shape, self.num_stats)
        :param kwargs: keyword arguments
               Depending on the implementation, kwargs can also contain further arguments.
        :return: argmax of the exponential family (Tensor).
                 Should be of shape (self.num_var, self.num_dims, *self.array_shape)
        """
