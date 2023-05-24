from typing import Dict

import numpy as np
import torch

from cirkit.einet.Layer import Layer


class FactorizedLeafLayer(Layer):
    """
    Computes all EiNet leaves in parallel, where each leaf is a vector of factorized distributions, where factors are
    from exponential families.

    In FactorizedLeafLayer, we generate an ExponentialFamilyArray with array_shape = (num_dist, num_replica), where
        num_dist is the vector length of the vectorized distributions (K in the paper), and
        num_replica is picked large enough such that "we compute enough leaf densities". At the moment we rely that
            the PC structure (see Class Graph) provides the necessary information to determine num_replica. In
            particular, we require that each leaf of the graph has the field einet_address.replica_idx defined;
            num_replica is simply the max over all einet_address.replica_idx.
            In the future, it would convenient to have an automatic allocation of leaves to replica, without requiring
            the user to specify this.
    The generate ExponentialFamilyArray has shape (batch_size, num_var, num_dist, num_replica). This array of densities
    will contain all densities over single RVs, which are then multiplied (actually summed, due to log-domain
    computation) together in forward(...).
    """

    def __init__(self, leaves, num_var, num_dims, exponential_family, ef_args, use_em=True):
        """
        :param leaves: list of PC leaves (DistributionVector, see Graph.py)
        :param num_var: number of random variables (int)
        :param num_dims: dimensionality of RVs (int)
        :param exponential_family: type of exponential family (derived from ExponentialFamilyArray)
        :param ef_args: arguments of exponential_family
        :param use_em: use on-board EM algorithm? (boolean)
        """
        super(FactorizedLeafLayer, self).__init__(use_em=use_em)

        self.nodes = leaves
        self.num_var = num_var
        self.num_dims = num_dims

        num_dist = list(set([n.num_dist for n in self.nodes]))
        if len(num_dist) != 1:
            raise AssertionError("All leaves must have the same number of distributions.")
        num_dist = num_dist[0]

        replica_indices = set([n.einet_address.replica_idx for n in self.nodes])
        if sorted(list(replica_indices)) != list(range(len(replica_indices))):
            raise AssertionError("Replica indices should be consecutive, starting with 0.")
        num_replica = len(replica_indices)

        # this computes an array of (batch, num_var, num_dist, num_repetition) exponential family densities
        # see ExponentialFamilyArray
        self.ef_array = exponential_family(num_var, num_dims, (num_dist, num_replica), use_em=use_em, **ef_args)

        # self.scope_tensor indicates which densities in self.ef_array belongs to which leaf.
        # TODO: it might be smart to have a sparse implementation -- I have experimented a bit with this, but it is not
        # always faster.
        self.register_buffer('scope_tensor', torch.zeros((num_var, num_replica, len(self.nodes))))
        for c, node in enumerate(self.nodes):
            self.scope_tensor[node.scope, node.einet_address.replica_idx, c] = 1.0
            node.einet_address.layer = self
            node.einet_address.idx = c

    def num_of_param(self) -> int:
        return int(np.prod(self.ef_array.params_shape))

    def get_shape_dict(self) -> Dict[str, int]:
        shape = self.ef_array.params_shape
        return dict(zip(("num_var", "num_dist", "num_replica", "num_stats"), shape))

    def freeze(self, freeze=True):
        for param in self.ef_array.parameters():
            param.requires_grad = not freeze
        self.frozen = freeze

    def is_frozen(self):
        return self.frozen

    # --------------------------------------------------------------------------------
    # Implementation of Layer interface

    def default_initializer(self):
        return self.ef_array.default_initializer()

    def initialize(self, initializer=None):
        self.ef_array.initialize(initializer)

    def forward(self, x=None):
        """
        Compute the factorized leaf densities. We are doing the computation in the log-domain, so this is actually
        computing sums over densities.

        We first pass the data x into self.ef_array, which computes a tensor of shape
        (batch_size, num_var, num_dist, num_replica). This is best interpreted as vectors of length num_dist, for each
        sample in the batch and each RV. Since some leaves have overlapping scope, we need to compute "enough" leaves,
        hence the num_replica dimension. The assignment of these log-densities to leaves is represented with
        self.scope_tensor.
        In the end, the factorization (sum in log-domain) is realized with a single einsum.

        :param x: input data (Tensor).
                  If self.num_dims == 1, this can be either of shape (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape (batch_size, self.num_var, self.num_dims).
        :return: log-density vectors of leaves
                 Will be of shape (batch_size, num_dist, len(self.nodes))
                 Note: num_dist is K in the paper, len(self.nodes) is the number of PC leaves
        """
        self.prob = torch.einsum('bxir,xro->bio', self.ef_array(x), self.scope_tensor)

        assert not torch.isnan(self.prob).any()
        assert not torch.isinf(self.prob).any()

    def backtrack(self, dist_idx, node_idx, mode='sample', **kwargs):
        """
        Backtrackng mechanism for EiNets.

        :param dist_idx: list of N indices into the distribution vectors, which shall be sampled.
        :param node_idx: list of N indices into the leaves, which shall be sampled.
        :param mode: 'sample' or 'argmax'; for sampling or MPE approximation, respectively.
        :param kwargs: keyword arguments
        :return: samples (Tensor). Of shape (N, self.num_var, self.num_dims).
        """
        if len(dist_idx) != len(node_idx):
            raise AssertionError("Invalid input.")

        with torch.no_grad():
            N = len(dist_idx)
            if mode == 'sample':
                ef_values = self.ef_array.sample(N, **kwargs)
            elif mode == 'argmax':
                ef_values = self.ef_array.argmax(**kwargs)
            else:
                raise AssertionError('Unknown backtracking mode {}'.format(mode))

            values = torch.zeros((N, self.num_var, self.num_dims), device=ef_values.device, dtype=ef_values.dtype)

            for n in range(N):
                cur_value = torch.zeros(self.num_var, self.num_dims, device=ef_values.device, dtype=ef_values.dtype)
                if len(dist_idx[n]) != len(node_idx[n]):
                    raise AssertionError("Invalid input.")
                for c, k in enumerate(node_idx[n]):
                    scope = list(self.nodes[k].scope)
                    rep = self.nodes[k].einet_address.replica_idx
                    if mode == 'sample':
                        cur_value[scope, :] = ef_values[n, scope, :, dist_idx[n][c], rep]
                    elif mode == 'argmax':
                        cur_value[scope, :] = ef_values[scope, :, dist_idx[n][c], rep]
                    else:
                        raise AssertionError('Unknown backtracking mode {}'.format(mode))
                values[n, :, :] = cur_value

            return values

    def em_set_hyperparams(self, online_em_frequency, online_em_stepsize, purge=True):
        self.ef_array.em_set_hyperparams(online_em_frequency, online_em_stepsize, purge)

    def em_purge(self):
        self.ef_array.em_purge()

    def em_process_batch(self):
        self.ef_array.em_process_batch()

    def em_update(self):
        self.ef_array.em_update()

    def project_params(self, params):
        self.ef_array.project_params(params)

    def reparam_function(self):
        return self.ef_array.reparam_function()

    # --------------------------------------------------------------------------------

    def set_marginalization_idx(self, idx):
        """Set indicices of marginalized variables."""
        self.ef_array.set_marginalization_idx(idx)

    def get_marginalization_idx(self):
        """Get indicices of marginalized variables."""
        return self.ef_array.get_marginalization_idx()
