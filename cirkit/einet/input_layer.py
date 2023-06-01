import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Type

import torch
from torch import Tensor

from cirkit.region_graph.rg_node import RegionNode

from .exp_family import ExponentialFamilyArray
from .layer import Layer


# TODO: but we don't have a non-factorized one
class FactorizedInputLayer(Layer):
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

    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]
        self,
        nodes: List[RegionNode],
        num_var: int,
        num_dims: int,
        exponential_family: Type[ExponentialFamilyArray],
        ef_args: Dict[str, Any],
        use_em: bool = True,
    ):
        """Init class.

        :param nodes: list of PC leaves (DistributionVector, see Graph.py)
        :param num_var: number of random variables (int)
        :param num_dims: dimensionality of RVs (int)
        :param exponential_family: type of exponential family (derived from ExponentialFamilyArray)
        :param ef_args: arguments of exponential_family
        :param use_em: use on-board EM algorithm? (boolean)
        """
        super().__init__(use_em=use_em)

        self.nodes = nodes
        self.num_var = num_var
        self.num_dims = num_dims

        num_dists = set(n.num_dist for n in self.nodes)
        assert len(num_dists) == 1, "All leaves must have the same number of distributions."
        num_dist = num_dists.pop()

        replica_indices = set(n.einet_address.replica_idx for n in self.nodes)
        num_replica = len(replica_indices)
        assert replica_indices == set(
            range(num_replica)
        ), "Replica indices should be consecutive, starting with 0."

        # this computes an array of (batch, num_var, num_dist, num_repetition)
        # exponential family densities
        # see ExponentialFamilyArray
        self.ef_array = exponential_family(
            num_var,
            num_dims,
            (num_dist, num_replica),
            use_em=use_em,
            **ef_args,  # type: ignore[misc]
        )

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

        self.frozen = False  # TODO: this is default right?

    def num_of_param(self) -> int:
        """Get number of params.

        Returns:
            int: The number of params.
        """
        return math.prod(self.ef_array.params_shape)

    def freeze(self, freeze: bool = True) -> None:
        """Freeze all params from bw.

        Args:
            freeze (bool, optional): Whether to freeze or unfreeze. Defaults to True.
        """
        for param in self.ef_array.parameters():
            param.requires_grad = not freeze
        self.frozen = freeze

    # TODO: do we need this? or use a property?
    def is_frozen(self) -> bool:
        """Test if is frozen.

        Returns:
            bool: Whether is frozen.
        """
        return self.frozen

    def get_shape_dict(self) -> Dict[str, int]:
        """Get the shape dict.

        Returns:
            Dict[str, int]: The shape dict.
        """
        return dict(
            zip(("num_var", "num_dist", "num_replica", "num_stats"), self.ef_array.params_shape)
        )

    def default_initializer(self) -> Tensor:
        """Init by default.

        Returns:
            Tensor: The default.
        """
        return self.ef_array.default_initializer()

    def initialize(self, initializer: Optional[Tensor] = None) -> None:
        """Init by given.

        Args:
            initializer (Optional[Tensor], optional): Given init. Defaults to None.
        """
        self.ef_array.initialize(
            initializer if isinstance(initializer, Tensor) else "default"  # type: ignore[misc]
        )

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
        self.prob = torch.einsum("bxir,xro->bio", self.ef_array(x), self.scope_tensor)

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
                self.ef_array.sample(big_n, **kwargs)  # type: ignore[misc]
                if mode == "sample"
                else self.ef_array.argmax(**kwargs)  # type: ignore[misc]
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

    # TODO: do we really need a layer of wrapper?
    def em_set_hyperparams(
        self, online_em_frequency: int, online_em_stepsize: float, purge: bool = True
    ) -> None:
        """Set new setting for online EM.

        Args:
            online_em_frequency (int): I don't know.
            online_em_stepsize (float): I don't know.
            purge (bool, optional): Whether to purge. Defaults to True.
        """
        self.ef_array.em_set_hyperparams(online_em_frequency, online_em_stepsize, purge)

    def em_purge(self) -> None:
        """Discard em statistics."""
        self.ef_array.em_purge()

    def em_process_batch(self) -> None:
        """Accumulate EM statistics of current batch.

        This should typically be called via EinsumNetwork.em_process_batch().
        """
        self.ef_array.em_process_batch()

    def em_update(self) -> None:
        """Do an EM update."""
        self.ef_array.em_update()

    def project_params(self, params: Tensor) -> None:
        """Project onto parameters' constraint set.

        Exponential families are usually defined on a constrained domain, e.g. \
            the second parameter of a Gaussian needs \
            to be non-negative. The EM algorithm takes the parameters sometimes \
            out of their domain. This function projects \
            them back onto their domain.

        :param params: the current parameters, same shape as self.params.
        """
        # TODO: why discard return?
        self.ef_array.project_params(params)

    def reparam_function(self) -> Tensor:
        """Re-parameterize parameters, in order that they stay in their constrained domain.

        When we are not using the EM, we need to transform unconstrained \
            (real-valued) parameters to the constrained set \
            of the expectation parameter. This function should return such a \
            function (i.e. the return value should not be \
            a projection, but a function which does the projection).

        :return: function object f which takes as input unconstrained parameters (Tensor) \
            and returns re-parametrized parameters.
        """
        # TODO: where is the params?
        return self.ef_array.reparam_function(torch.Tensor())

    def set_marginalization_idx(self, idx: Tensor) -> None:
        """Set indicices of marginalized variables.

        Args:
            idx (Tensor): The indices.
        """
        self.ef_array.set_marginalization_idx(idx)

    # TODO: why optional?
    def get_marginalization_idx(self) -> Optional[Tensor]:
        """Get indicices of marginalized variables.

        Returns:
            Tensor: The indices.
        """
        return self.ef_array.get_marginalization_idx()
