import functools
import warnings
from itertools import count
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from cirkit.einet._Layer import Layer
from tensorly import set_backend
from torch.nn.functional import softmax
from cirkit.einet._utils import sample_matrix_categorical


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
