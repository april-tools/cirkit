import functools
import warnings
from itertools import count
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from cirkit.einet.Layer import Layer
from tensorly import set_backend
from torch.nn.functional import softmax
from utils import sample_matrix_categorical


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


set_backend('pytorch')


class SumLayer(Layer):
    """
    Implements an abstract SumLayer class. Takes care of parameters and EM.
    EinsumLayer and MixingLayer are derived from SumLayer.
    """

    def __init__(self): # TODO: restore params mask ", params_mask=None):"
        """
        :param params_shape: shape of tensor containing all sum weights (tuple of ints).
        :param normalization_dims: the dimensions (axes) of the sum-weights which shall be normalized
                                   (int of tuple of ints)
        :param use_em: use the on-board EM algorithm?
        :param params_mask: binary mask for masking out certain parameters (tensor of shape params_shape).
        """
        super(SumLayer, self).__init__()
        self.frozen = False
    # --------------------------------------------------------------------------------
    # The following functions need to be implemented in derived classes.

    def num_of_param(self) -> int:
        raise NotImplementedError

    def freeze(self, freeze=True):
        for param in self.parameters():
            param.requires_grad = not freeze
        self.frozen = freeze

    def is_frozen(self):
        return self.frozen

    def _forward(self, params=None):
        """
        Implementation of the actual sum operation.

        :param params: sum-weights to use.
        :return: result of the sum layer. Must yield a (batch_size, num_dist, num_nodes) tensor of log-densities.
                 Here, num_dist is the vector length of vectorized sums (K in the paper), and num_nodes is the number
                 of sum nodes in this layer.
        """
        raise NotImplementedError

    def _backtrack(self, dist_idx, node_idx, sample_idx, params, use_evidence=False, mode='sample', **kwargs):
        """
        Helper routine to implement EiNet backtracking, for sampling or MPE approximation.

        dist_idx, node_idx, sample_idx are lists of indices, all of the same length.

        :param dist_idx: list of indices, indexing into vectorized sums.
        :param node_idx: list of indices, indexing into node list of this layer.
        :param sample_idx: list of sample indices; representing the identity of the samples the EiNet is about to
                           generate. We need this, since not every SumLayer necessarily gets selected in the top-down
                           sampling process.
        :param params: sum-weights to use (Tensor).
        :param use_evidence: incorporate the bottom-up evidence (Bool)? For conditional sampling.
        :param mode: 'sample' or 'argmax'; for sampling or MPE approximation, respectively.
        :param kwargs: Additional keyword arguments.
        :return: depends on particular implementation.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------------------
    def initialize(self, initializer='default'):
        raise NotImplementedError

    def forward(self, x=None):
        """
        Evaluate this SumLayer.

        :param x: unused
        :return: tensor of log-densities. Must be of shape (batch_size, num_dist, num_nodes).
                 Here, num_dist is the vector length of vectorized sum nodes (K in the paper), and num_nodes is the
                 number of sum nodes in this layer.
        """
        # TODO the distinction em or not has not to be done here
        """ if self._use_em:
            params = self.params
        else:
            reparam = self.reparam(self.params)
            params = reparam 
        """
        # params = self.params
        self._forward()

    def backtrack(self, dist_idx, node_idx, sample_idx, use_evidence=False, mode='sample', **kwargs):

        raise NotImplementedError
        """
        Helper routine for backtracking in EiNets, see _sample(...) for details.
        """

        if mode != 'sample' and mode != 'argmax':
            raise AssertionError('Unknown backtracking mode {}'.format(mode))

        if self._use_em:
            params = self.params
        else:
            with torch.no_grad():
                params = self.reparam(self.params)
        return self._backtrack(dist_idx, node_idx, sample_idx, params, use_evidence, mode, **kwargs)

    @deprecated
    def em_purge(self):
        raise NotImplementedError
        """ Discard em statistics."""
        if self.params is not None:
            self.params.grad = None

    @deprecated
    def em_process_batch(self):
        raise NotImplementedError
        """
        Accumulate EM statistics of current batch. This should be called after call to backwards() on the output of
        the EiNet.
        """
        if not self._use_em:
            raise AssertionError("em_process_batch called while _use_em==False.")
        if self.params is None:
            return

        if self.online_em_frequency is not None:
            self._online_em_counter += 1
            if self._online_em_counter == self.online_em_frequency:
                self.em_update(True)
                self._online_em_counter = 0

    @deprecated
    def em_update(self, _triggered=False):
        raise NotImplementedError

        """
        Do an EM update. If the setting is online EM (online_em_stepsize is not None), then this function does nothing,
        since updates are triggered automatically. Thus, leave the private parameter _triggered alone.

        :param _triggered: for internal use, don't set
        :return: None
        """
        if not self._use_em:
            raise AssertionError("em_update called while _use_em==False.")
        if self.params is None:
            return

        if self.online_em_stepsize is not None and not _triggered:
            return

        with torch.no_grad():
            n = self.params.grad * self.params.data

            if self.online_em_stepsize is None:
                self.params.data = n
            else:
                s = self.online_em_stepsize
                p = torch.clamp(n, 1e-16)
                p = p / (p.sum(self.normalization_dims, keepdim=True))
                self.params.data = (1. - s) * self.params + s * p

            self.params.data = torch.clamp(self.params, 1e-16)
            if self.params_mask is not None:
                self.params.data *= self.params_mask
            self.params.data = self.params / (self.params.sum(self.normalization_dims, keepdim=True))
            self.params.grad = None

    def reparam_function(self):
        """
        Reparametrization function, transforming unconstrained parameters into valid sum-weight
        (non-negative, normalized).
        """

        def reparam(params_in):
            other_dims = tuple(i for i in range(len(params_in.shape)) if i not in self.normalization_dims)

            permutation = other_dims + self.normalization_dims
            unpermutation = tuple(c for i in range(len(permutation)) for c, j in enumerate(permutation) if j == i)

            numel = functools.reduce(lambda x, y: x * y, [params_in.shape[i] for i in self.normalization_dims])

            other_shape = tuple(params_in.shape[i] for i in other_dims)
            params_in = params_in.permute(permutation)
            orig_shape = params_in.shape
            params_in = params_in.reshape(other_shape + (numel,))
            out = softmax(params_in, -1)
            out = out.reshape(orig_shape).permute(unpermutation)
            return out

        return reparam

    @deprecated
    def project_params(self):
        """Currently not required."""
        raise NotImplementedError


class EinsumMixingLayer(SumLayer):
    """
    Implements the Mixing Layer, in order to handle sum nodes with multiple children.
    Recall Figure II from above:

           S          S
        /  |  \      / \
       P   P  P     P  P
      /\   /\  /\  /\  /\
     N  N N N N N N N N N

    Figure II


    We implement such excerpt as in Figure III, splitting sum nodes with multiple children in a chain of two sum nodes:

            S          S
        /   |  \      / \
       S    S   S    S  S
       |    |   |    |  |
       P    P   P    P  P
      /\   /\  /\  /\  /\
     N  N N N N N N N N N

    Figure III


    The input nodes N have already been computed. The product nodes P and the first sum layer are computed using an
    EinsumLayer, yielding a log-density tensor of shape
        (batch_size, vector_length, num_nodes).
    In this example num_nodes is 5, since the are 5 product nodes (or 5 singleton sum nodes). The EinsumMixingLayer
    then simply mixes sums from the first layer, to yield 2 sums. This is just an over-parametrization of the original
    excerpt.
    """

    def __init__(self, graph, nodes, einsum_layer):
        """
        :param graph: the PC graph (see Graph.py)
        :param nodes: the nodes of the current layer (see constructor of EinsumNetwork), which have multiple children
        :param einsum_layer:
        :param use_em:
        """
        self.nodes = nodes

        self.num_sums = set([n.num_dist for n in self.nodes])
        if len(self.num_sums) != 1:
            raise AssertionError("Number of distributions must be the same for all regions in one layer.")
        self.num_sums = list(self.num_sums)[0]

        self.max_components = max([len(graph.succ[n]) for n in self.nodes])
        # einsum_layer is actually the only layer which gives input to EinsumMixingLayer
        # we keep it in a list, since otherwise it gets registered as a torch sub-module
        self.layers = [einsum_layer]
        self.mixing_component_idx = einsum_layer.mixing_component_idx

        if einsum_layer.dummy_idx is None:
            raise AssertionError('EinsumLayer has not set a dummy index for padding.')

        param_shape = (self.num_sums, len(self.nodes), self.max_components)
        # param_shape = (len(self.nodes), self.max_components) for better perf

        # The following code does some bookkeeping.
        # padded_idx indexes into the log-density tensor of the previous EinsumLayer, padded with a dummy input which
        # outputs constantly 0 (-inf in the log-domain), see class EinsumLayer.
        padded_idx = []
        params_mask = torch.ones(param_shape)
        for c, node in enumerate(self.nodes):
            num_components = len(self.mixing_component_idx[node])
            padded_idx += self.mixing_component_idx[node]
            padded_idx += [einsum_layer.dummy_idx] * (self.max_components - num_components)
            if self.max_components > num_components:
                params_mask[:, c, num_components:] = 0.0
            node.einet_address.layer = self
            node.einet_address.idx = c

        super(EinsumMixingLayer, self).__init__()

        ####### CODE ORIGINALLY FROM SUMLAYER
        self.params_shape = param_shape
        self.params = None
        self.normalization_dims = (2,)
        self.register_buffer('params_mask', params_mask)
        ############## END

        self.register_buffer('padded_idx', torch.tensor(padded_idx))
        self.softmax: bool = None

    def num_of_param(self) -> int:
        return int(np.prod(self.params_shape))

    def project_params(self):
        raise NotImplementedError

    @property
    def clamp_value(self) -> float:
        return torch.finfo(self.params.data.dtype).smallest_normal

    def clamp_params(self, all=False):
        """
        Clamp parameters such that they are non-negative and
        is impossible to get zero probabilities.
        This involves using a constant that is specific on the computation
        :return:
        """
        if not all:
            if self.params.requires_grad:
                self.params.data.clamp_(min=self.clamp_value)
        else:
            self.params.data.clamp_(min=self.clamp_value)

    def default_initializer(self):
        """
        A simple initializer for normalized sum-weights.
        :return: initial parameters
        """
        if self.softmax:
            raise NotImplementedError
            params = torch.rand(self.params_shape)
        else:
            params = 0.01 + 0.98 * torch.rand(self.params_shape)
        #assert torch.all(params >= 0)

        with torch.no_grad():
            if self.params_mask is not None:
                params.data *= self.params_mask

            if not self.softmax:
                params.data = params.data / (params.data.sum(self.normalization_dims, keepdim=True))

        #assert torch.all(params >= 0)
        return params

    def initialize(self, initializer='default'):
        """
        Initialize the parameters for this SumLayer.

        :param initializer: denotes the initialization method.
               If 'default' (str): use the default initialization, and store the parameters locally.
               If Tensor: provide custom initial parameters.
        :return: None
        """
        assert initializer is not None

        if type(initializer) == str and initializer == 'default':
            self.params = torch.nn.Parameter(self.default_initializer())
        elif type(initializer) == torch.Tensor:
            if initializer.shape != self.params_shape:
                raise AssertionError("Incorrect parameter shape.")
            self.params = torch.nn.Parameter(initializer)
        else:
            raise AssertionError("Unknown initializer.")

    def get_parameters(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.params

    def _forward(self, params=None):

        self.child_log_prob = self.layers[0].prob[:, :, self.padded_idx]
        self.child_log_prob = self.child_log_prob.reshape((self.child_log_prob.shape[0],
                                                           self.child_log_prob.shape[1],
                                                           len(self.nodes),
                                                           self.max_components))

        max_p = torch.max(self.child_log_prob, 3, keepdim=True)[0]
        prob = torch.exp(self.child_log_prob - max_p)

        if self.softmax:
            params = softmax(self.params, -1)
        else:
            params = self.params

        assert torch.eq(self.params * self.params_mask, self.params).all()

        output = torch.einsum('bonc,onc->bon', prob, params)
        self.prob = torch.log(output) + max_p[:, :, :, 0]

        if torch.isnan(self.prob).any():
            assert not torch.isnan(self.prob).any()

        if torch.isinf(self.prob).any():
            assert not torch.isinf(self.prob).any()

    def _backtrack(self, dist_idx, node_idx, sample_idx, params, use_evidence=False, mode='sample', **kwargs):
        """Helper routine for backtracking in EiNets."""
        with torch.no_grad():
            if use_evidence:
                log_prior = torch.log(params[dist_idx, node_idx, :])
                log_posterior = log_prior + self.child_log_prob[sample_idx, dist_idx, node_idx, :]
                posterior = torch.exp(log_posterior - torch.logsumexp(log_posterior, 1, keepdim=True))
            else:
                posterior = params[dist_idx, node_idx, :]

            if mode == 'sample':
                idx = sample_matrix_categorical(posterior)
            elif mode == 'argmax':
                idx = torch.argmax(posterior, -1)
            dist_idx_out = dist_idx
            node_idx_out = [self.mixing_component_idx[self.nodes[i]][idx[c]] for c, i in enumerate(node_idx)]
            layers_out = [self.layers[0]] * len(node_idx)

        return dist_idx_out, node_idx_out, layers_out


class GenericEinsumLayer(SumLayer):

    # we have to provide operation for input, operation for product and operation after product
    #
    def __init__(self, graph, products, layers, prod_exp: bool, k: int):

        # self.r = r
        self.products = products
        # self.exp_reparam: bool = None
        self.prod_exp = prod_exp

        # assert decomposition_strategy in ["slice", "full"]
        # self.decomposition_strategy = decomposition_strategy

        # # # # # # # # #
        #   CHECK
        # # # # # # # # #

        set_num_sums = set([n.num_dist for p in self.products for n in graph.pred[p]])
        if len(set_num_sums) != 1:
            raise AssertionError("Number of distributions must be the same for all parent nodes in one layer.")
        # check if it is root
        num_sums_from_graph = list(set_num_sums)[0]
        if num_sums_from_graph > 1:
            self.num_sums = k
            # set num_sums in the graph
            successors = set([n for p in self.products for n in graph.pred[p]])
            for n in successors:
                n.num_dist = k

        else:
            self.num_sums = num_sums_from_graph

        self.num_input_dist = set([n.num_dist for p in self.products for n in graph.succ[p]])
        if len(self.num_input_dist) != 1:
            raise AssertionError(f"Number of input distributions must be the same for all child nodes in one layer. {self.num_input_dist}")
        self.num_input_dist = list(self.num_input_dist)[0]

        if any([len(graph.succ[p]) != 2 for p in self.products]):
            raise AssertionError("Only 2-partitions are currently supported.")

        # # # # # # # # #
        #   BUILD
        # # # # # # # # #
        super(GenericEinsumLayer, self).__init__()

        # MAKE PARAMETERS
        self.params_dict, self.shape_dict = self.build_params()
        self.params_mask = None # TODO: check usage for this

        # get pairs of nodes which are input to the products (list of lists)
        # length of the outer list is same as self.products, length of inner lists is 2
        # "left child" has index 0, "right child" has index 1
        self.inputs = [sorted(graph.successors(p)) for p in self.products]

        # collect all layers which contain left/right children
        self.left_layers = [l for l in layers if any([i[0].einet_address.layer == l for i in self.inputs])]
        self.right_layers = [l for l in layers if any([i[1].einet_address.layer == l for i in self.inputs])]

        # The following code does some index bookkeeping, in order that we can gather the required data in forward(...).
        # Recall that in EiNets, each layer implements a log-density tensor of shape
        # (batch_size, vector_length, num_nodes).
        # We iterate over all previous left/right layers, and collect the node indices (corresponding to axis 2) in
        # self.idx_layer_i_child_j, where i indexes into self.left_layers for j==0, and into self.left_layers for j==1.
        # These index collections allow us to simply iterate over the previous layers and extract the required node
        # slices in forward.
        #
        # Furthermore, the following code also generates self.permutation_child_0 and self.permutation_child_1,
        # which are permutations of the left and right input nodes. We need this to get them in the same order as
        # assumed in self.products.
        def do_input_bookkeeping(layers, child_num):
            permutation = [None] * len(self.inputs)
            permutation_counter = count(0)
            for layer_counter, layer in enumerate(layers):
                cur_idx = []
                for c, input in enumerate(self.inputs):
                    if input[child_num].einet_address.layer == layer:
                        cur_idx.append(input[child_num].einet_address.idx)
                        if permutation[c] is not None:
                            raise AssertionError("This should not happen.")
                        permutation[c] = next(permutation_counter)
                self.register_buffer('idx_layer_{}_child_{}'.format(layer_counter, child_num), torch.tensor(cur_idx))
            if any(i is None for i in permutation):
                raise AssertionError("This should not happen.")
            self.register_buffer('permutation_child_{}'.format(child_num), torch.tensor(permutation))

        do_input_bookkeeping(self.left_layers, 0)
        do_input_bookkeeping(self.right_layers, 1)

        # when the EinsumLayer is followed by a EinsumMixingLayer, we produce a dummy "node" which outputs 0 (-inf in
        # log-domain) for zero-padding.
        self.dummy_idx = None

        # the dictionary mixing_component_idx stores which nodes (axis 2 of the log-density tensor) need to get mixed
        # in the following EinsumMixingLayer
        self.mixing_component_idx = {}

        for c, product in enumerate(self.products):
            # each product must have exactly 1 parent (sum node)
            node = list(graph.predecessors(product))
            assert len(node) == 1
            node = node[0]

            if len(graph.succ[node]) == 1:
                node.einet_address.layer = self
                node.einet_address.idx = c
            else:  # case followed by EinsumMixingLayer
                if node not in self.mixing_component_idx:
                    self.mixing_component_idx[node] = []
                self.mixing_component_idx[node].append(c)
                self.dummy_idx = len(self.products)

    @property
    def clamp_value(self) -> float:
        """
        Value for parameters clamping to keep all probabilities greater than 0.
        :return: value for parameters clamping
        """
        raise NotImplementedError

    def clamp_params(self, all=False):
        """
        Clamp parameters such that they are non-negative and
        is impossible to get zero probabilities.
        This involves using a constant that is specific on the computation
        :return:
        """
        for name, par in self.get_params_dict().items():
            if not all:
                if par.requires_grad:
                    par.data.clamp_(min=self.clamp_value)
            else:
                par.data.clamp_(min=self.clamp_value)

    def build_params(self) -> (Dict[str, Any], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        raise NotImplementedError

    #def project_params(self): TODO: check is needed?
    #    raise NotImplementedError

    def initialize(self, initializer: Union[str, Tuple[torch.Tensor]] = "default"):
        """
        Initialize the layer parameters.
        :param initializer: Unused
        :return:
        """
        def random_nonneg_tensor(shape: Tuple[int, ...]):
            return 0.01 + 0.98 * torch.rand(shape)

        for key in self.params_dict:
            self.__setattr__(key, torch.nn.Parameter(random_nonneg_tensor(self.shape_dict[key])))
            self.params_dict[key] = self.__getattr__(key)

    @property
    def params_shape(self):
        whole_shape: Tuple[Tuple[int], ...] = ()
        for key, value in self.shape_dict.items():
            whole_shape += (value,)

        return whole_shape

    def get_params_dict(self) -> Dict[str, torch.nn.Parameter]:
        assert self.params_dict is not None
        return self.params_dict

    def num_of_param(self):
        """
        Returns the total number of parameters of the layer
        :return: the total number of parameters of the layer
        """
        param_acc = 0
        for key, value in self.shape_dict.items():
            param_acc += np.prod(value)

        return int(param_acc)

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        """
        Computes the main Einsum operation of the layer.

        :param left_prob: value in log space for left child
        :param right_prob: value in log space for right child
        :return: result of the left operations, in log-space
        """
        raise NotImplementedError

    def _forward(self, params=None):
        """
        EinsumLayer forward pass.
        We assume that all parameters are in the correct range (no checks done).

        Skeleton for each EinsumLayer (options Xa and Xb are mutual exclusive and follows an a-path o b-path)
        1) Go To exp-space (with maximum subtraction) -> NON SPECIFIC
        2a) Do the einsum operation and go to the log space || 2b) Do the einsum operation
        3a) do the sum                                      || 3b) do the product
        4a) go to exp space do the einsum and back to log   || 4b) do the einsum operation [OPTIONAL]
        5a) do nothing                                      || 5b) back to log space
        """

        def cidx(layer_counter, child_num):
            return self.__getattr__('idx_layer_{}_child_{}'.format(layer_counter, child_num))

        # iterate over all layers which contain "left" nodes, get their indices; then, concatenate them to one tensor
        self.left_child_log_prob = torch.cat([l.prob[:, :, cidx(c, 0)] for c, l in enumerate(self.left_layers)], 2)
        # get into the same order as assumed in self.products
        self.left_child_log_prob = self.left_child_log_prob[:, :, self.permutation_child_0]
        # ditto, for right "right" nodes
        self.right_child_log_prob = torch.cat([l.prob[:, :, cidx(c, 1)] for c, l in enumerate(self.right_layers)], 2)
        self.right_child_log_prob = self.right_child_log_prob[:, :, self.permutation_child_1]

        assert not torch.isinf(self.left_child_log_prob).any()
        assert not torch.isinf(self.right_child_log_prob).any()
        assert not torch.isnan(self.left_child_log_prob).any()
        assert not torch.isnan(self.right_child_log_prob).any()

        # # # # # # # # # # STEP 1: Go To the exp space # # # # # # # # # #
        # We perform the LogEinsumExp trick, by first subtracting the maxes
        log_prob = self.central_einsum(self.left_child_log_prob, self.right_child_log_prob)

        if torch.isinf(log_prob).any():
            raise AssertionError("Inf log prob")
        if torch.isnan(log_prob).any():
            raise AssertionError("NaN log prob")

        # zero-padding (-inf in log-domain) for the following mixing layer
        if self.dummy_idx:
            log_prob = F.pad(log_prob, [0, 1], "constant", float('-inf'))

        self.prob = log_prob

    def _backtrack(self, dist_idx, node_idx, sample_idx, params, use_evidence=False, mode='sample', **kwargs):
        raise NotImplementedError # TODO: implement backtrack


class CPEinsumLayer(GenericEinsumLayer):
    def __init__(self, graph, products, layers, k: int, prod_exp=False, r=1):
        self.r = r
        super(CPEinsumLayer, self).__init__(graph, products, layers, prod_exp, k=k)

    def build_params(self) -> (Dict[str, Union[None, torch.nn.Parameter]], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        params_dict = {"cp_a": None, "cp_b": None, "cp_c": None}
        shapes_dict = {"cp_a": (self.num_input_dist, self.r, len(self.products)),
                       "cp_b": (self.num_input_dist, self.r, len(self.products)),
                       "cp_c": (self.num_sums, self.r, len(self.products))}
        return params_dict, shapes_dict

    @property
    def clamp_value(self) -> float:
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal

        if self.prod_exp:
            return torch.pow(torch.Tensor([smallest_normal]), 1/3).item()
        else:
            return torch.sqrt(torch.Tensor([smallest_normal])).item()

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        pa = self.params_dict["cp_a"]
        pb = self.params_dict["cp_b"]
        pc = self.params_dict["cp_c"]

        left_hidden = torch.einsum('bip,irp->brp', left_prob, pa)
        right_hidden = torch.einsum('bjp,jrp->brp', right_prob, pb)

        if self.prod_exp:
            hidden = left_hidden * right_hidden
            prob = torch.einsum('brp,orp->bop', hidden, pc)
            log_prob = torch.log(prob) + left_max + right_max
        else:
            log_left_hidden = torch.log(left_hidden) + left_max
            log_right_hidden = torch.log(right_hidden) + right_max
            log_hidden = log_left_hidden + log_right_hidden

            hidden_max = torch.max(log_hidden, 1, keepdim=True)[0]
            hidden = torch.exp(log_hidden - hidden_max)
            prob = torch.einsum('brp,orp->bop', hidden, pc)
            log_prob = torch.log(prob) + hidden_max

        return log_prob


class CPSharedEinsumLayer(GenericEinsumLayer): # TODO edit this because of numerical stability
    def __init__(self, graph, products, layers, k: int, prod_exp=False, r=1):
        self.r = r
        super(CPSharedEinsumLayer, self).__init__(graph, products, layers, prod_exp, k=k)

    def build_params(self) -> (Dict[str, Union[None, torch.nn.Parameter]], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        params_dict = {"cp_a": None, "cp_b": None, "cp_c": None, "cp_d": None}
        shapes_dict = {"cp_a": (self.num_input_dist, self.r),
                       "cp_b": (self.num_input_dist, self.r),
                       "cp_c": (self.num_sums, self.r),
                       "cp_d": (len(self.products), self.r)}

        return params_dict, shapes_dict

    @property
    def clamp_value(self) -> float:
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal

        if self.prod_exp:
            return torch.pow(torch.Tensor([smallest_normal]), 1/4).item()
        else:
            return torch.sqrt(torch.Tensor([smallest_normal])).item()

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        pa = self.params_dict["cp_a"]
        pb = self.params_dict["cp_b"]
        pc = self.params_dict["cp_c"]
        pd = self.params_dict["cp_d"]

        left_hidden = torch.einsum('bip,ir->brp', left_prob, pa)
        right_hidden = torch.einsum('bjp,jr->brp', right_prob, pb)

        if self.prod_exp:
            hidden = left_hidden * right_hidden
            rescaled_hidden = torch.einsum('brp,pr->brp', hidden, pd)
            prob = torch.einsum('brp,or->bop', rescaled_hidden, pc)
            log_prob = torch.log(prob) + left_max + right_max
        else:
            log_left_hidden = torch.log(left_hidden) + left_max
            log_right_hidden = torch.log(right_hidden) + right_max
            log_hidden = log_left_hidden + log_right_hidden

            rescaled_log_hidden = log_hidden + torch.t(pd)
            hidden_max = torch.max(rescaled_log_hidden, 1, keepdim=True)[0]
            rescaled_hidden = torch.exp(rescaled_log_hidden - hidden_max)
            prob = torch.einsum('brp,or->bop', rescaled_hidden, pc)
            log_prob = torch.log(prob) + hidden_max

        return log_prob


class HCPTEinsumLayer(GenericEinsumLayer):
    def __init__(self, graph, products, layers, k: int, prod_exp=False):
        super(HCPTEinsumLayer, self).__init__(graph, products, layers, prod_exp, k=k)

    def build_params(self) -> (Dict[str, Union[None, torch.nn.Parameter]], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        params_dict = {"cp_a": None, "cp_b": None}
        shapes_dict = {"cp_a": (self.num_input_dist, self.num_sums, len(self.products)),
                       "cp_b": (self.num_input_dist, self.num_sums, len(self.products))}

        return params_dict, shapes_dict

    @property
    def clamp_value(self) -> float:
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal

        if self.prod_exp:
            return torch.sqrt(torch.Tensor([smallest_normal])).item()
        else:
            return torch.sqrt(torch.Tensor([smallest_normal])).item()# smallest_normal

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        pa = self.params_dict["cp_a"]
        pb = self.params_dict["cp_b"]

        left_hidden = torch.einsum('bip,irp->brp', left_prob, pa)
        right_hidden = torch.einsum('bjp,jrp->brp', right_prob, pb)

        if self.prod_exp:
            hidden = left_hidden * right_hidden
            log_prob = torch.log(hidden) + left_max + right_max
        else:
            log_left_hidden = torch.log(left_hidden) + left_max
            log_right_hidden = torch.log(right_hidden) + right_max
            log_prob = log_left_hidden + log_right_hidden

        return log_prob


class HCPTSharedEinsumLayer(GenericEinsumLayer):
    def __init__(self, graph, products, layers, k: int, prod_exp=False):
        super(HCPTSharedEinsumLayer, self).__init__(graph, products, layers, prod_exp, k=k)

    def build_params(self) -> (Dict[str, Union[None, torch.nn.Parameter]], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        params_dict = {"cp_a": None, "cp_b": None, "cp_d": None}
        shapes_dict = {"cp_a": (self.num_input_dist, self.num_sums),
                       "cp_b": (self.num_input_dist, self.num_sums),
                       "cp_d": (len(self.products), self.num_sums)}

        return params_dict, shapes_dict

    @property
    def clamp_value(self) -> float:
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal

        if self.prod_exp:
            return torch.sqrt(torch.Tensor([smallest_normal])).item()
        else:
            return torch.sqrt(torch.Tensor([smallest_normal])).item()

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        pa = self.params_dict["cp_a"]
        pb = self.params_dict["cp_b"]
        pd = self.params_dict["cp_d"]

        left_hidden = torch.einsum('bip,ir->brp', left_prob, pa)
        right_hidden = torch.einsum('bjp,jr->brp', right_prob, pb)

        if self.prod_exp:
            hidden = left_hidden * right_hidden
            log_prob = torch.log(hidden) + left_max + right_max
        else:
            log_left_hidden = torch.log(left_hidden) + left_max
            log_right_hidden = torch.log(right_hidden) + right_max
            log_prob = log_left_hidden + log_right_hidden

        log_prob = log_prob + torch.t(pd)

        return log_prob


class HCPTLoLoEinsumLayer(GenericEinsumLayer):
    def __init__(self, graph, products, layers, k: int, prod_exp=False, r=1):
        self.r = r
        super(HCPTLoLoEinsumLayer, self).__init__(graph, products, layers, prod_exp, k=k)

    def build_params(self) -> (Dict[str, Union[None, torch.nn.Parameter]], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        params_dict = {"cp_a1": None, "cp_a2": None, "cp_b1": None, "cp_b2": None}
        shapes_dict = {"cp_a1": (self.num_input_dist, self.r, len(self.products)),
                       "cp_a2": (self.r, self.num_sums, len(self.products)),
                       "cp_b1": (self.num_input_dist, self.r, len(self.products)),
                       "cp_b2": (self.r, self.num_sums, len(self.products))}

        return params_dict, shapes_dict

    @property
    def clamp_value(self) -> float:
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal

        if self.prod_exp:
            return torch.pow(torch.Tensor([smallest_normal]), 1/4).item()
        else:
            return torch.sqrt(torch.Tensor([smallest_normal])).item()

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        pa_1 = self.params_dict["cp_a1"]
        pa_2 = self.params_dict["cp_a2"]
        pb_1 = self.params_dict["cp_b1"]
        pb_2 = self.params_dict["cp_b2"]

        left_hidden = torch.einsum('bip,irp->brp', left_prob, pa_1)
        left_hidden = torch.einsum('brp,rip->bip', left_hidden, pa_2)
        right_hidden = torch.einsum('bjp,jrp->brp', right_prob, pb_1)
        right_hidden = torch.einsum('brp,rjp->bjp', right_hidden, pb_2)

        if self.prod_exp:
            hidden = left_hidden * right_hidden
            log_prob = torch.log(hidden) + left_max + right_max
        else:
            # LogEinsumExp trick, re-add the max
            log_left_hidden = torch.log(left_hidden) + left_max
            log_right_hidden = torch.log(right_hidden) + right_max
            log_prob = log_left_hidden + log_right_hidden

        return log_prob


class HCPTLoLoSharedEinsumLayer(GenericEinsumLayer): # TODO: same thing for numerical stability
    def __init__(self, graph, products, layers, k: int, prod_exp=False, r=1):
        self.r = r
        super(HCPTLoLoSharedEinsumLayer, self).__init__(graph, products, layers, prod_exp=prod_exp, k=k)

    def build_params(self) -> (Dict[str, Union[None, torch.nn.Parameter]], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        params_dict = {"cp_a1": None, "cp_a2": None, "cp_b1": None, "cp_b2": None, "cp_d1": None, "cp_d2": None}
        shapes_dict = {"cp_a1": (self.num_input_dist, self.r),
                       "cp_a2": (self.r, self.num_sums),
                       "cp_b1": (self.num_input_dist, self.r),
                       "cp_b2": (self.r, self.num_sums),
                       "cp_d1": (len(self.products), self.r),
                       "cp_d2": (len(self.products), self.r)}

        return params_dict, shapes_dict

    @property
    def clamp_value(self) -> float:
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal

        if self.prod_exp:
            return torch.pow(torch.Tensor([smallest_normal]), 1/4).item()
        else:
            return torch.pow(torch.Tensor([smallest_normal]), 1/3).item()

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        pa_1 = self.params_dict["cp_a1"]
        pa_2 = self.params_dict["cp_a2"]
        pb_1 = self.params_dict["cp_b1"]
        pb_2 = self.params_dict["cp_b2"]
        pd_1 = self.params_dict["cp_d1"]
        pd_2 = self.params_dict["cp_d2"]

        left_hidden = torch.einsum('bip,ir->brp', left_prob, pa_1)
        left_hidden = torch.einsum('brp,pr->brp', left_hidden, pd_1)
        left_hidden = torch.einsum('brp,ri->bip', left_hidden, pa_2)

        right_hidden = torch.einsum('bjp,jr->brp', right_prob, pb_1)
        right_hidden = torch.einsum('brp,pr->brp', right_hidden, pd_2)
        right_hidden = torch.einsum('brp,rj->bjp', right_hidden, pb_2)

        if self.prod_exp:
            hidden = left_hidden * right_hidden
            log_prob = torch.log(hidden) + left_max + right_max
        else:
            clamp_left_hidden = torch.log(left_hidden)
            clamp_right_hidden = torch.log(right_hidden)
            assert not torch.isinf(clamp_left_hidden).any()
            assert not torch.isinf(clamp_right_hidden).any()

            # LogEinsumExp trick, re-add the max
            log_left_hidden = clamp_left_hidden + left_max
            log_right_hidden = clamp_right_hidden + right_max
            log_prob = log_left_hidden + log_right_hidden

        return log_prob

#class HCPTSharedLoLoEinsumLayer(GenericEinsumLayer):
#    def __init__(self, graph, products, layers, decomposition_strategy="slice", prod_exp=False, r=1):
#        super(HCPTSharedLoLoEinsumLayer, self).__init__(graph, products, layers, decomposition_strategy, prod_exp, r)
#        self.prod_exp = prod_exp
#        self.r = r


class RescalEinsumLayer(GenericEinsumLayer):
    def __init__(self, graph, products, layers, k: int):
        super(RescalEinsumLayer, self).__init__(graph, products, layers, prod_exp=False, k=k)

    def build_params(self) -> (Dict[str, Union[None, torch.nn.Parameter]], Dict[str, Tuple[int, ...]]):
        """
        Create params dict for the layer (the parameters are uninitialized)
        :return: dict {params_name, params}
        """
        params_dict = {"params": None}
        shapes_dict = {"params": (self.num_input_dist, self.num_input_dist, self.num_sums, len(self.products))}

        return params_dict, shapes_dict

    @property
    def clamp_value(self) -> float:
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal

        return torch.sqrt(torch.Tensor([smallest_normal])).item()

    def central_einsum(self, left_prob: torch.Tensor, right_prob: torch.Tensor) -> torch.Tensor:
        left_max = torch.max(self.left_child_log_prob, 1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max = torch.max(self.right_child_log_prob, 1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        params = self.params_dict["params"]

        output = torch.einsum('bip,bjp,ijop->bop', left_prob, right_prob, params)
        log_prob = torch.log(output) + left_max + right_max

        return log_prob
