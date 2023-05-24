import warnings

import numpy as np
import torch

from cirkit.einet.EinsumLayer import (
    CPEinsumLayer,
    CPSharedEinsumLayer,
    EinsumMixingLayer,
    HCPTEinsumLayer,
    HCPTLoLoEinsumLayer,
    HCPTLoLoSharedEinsumLayer,
    HCPTSharedEinsumLayer,
    RescalEinsumLayer,
)
from cirkit.einet.leaf_layer import FactorizedLeafLayer
from cirkit.region_graph import graph as Graph

LAYER_TYPES = ["hcpt-lolo-shared", "hcpt-lolo", "rescal", "cp-shared", "hcpt", "hcpt-shared", "cp"]


class Args(object):
    """
    Arguments for EinsumNetwork class.

    num_var: number of random variables (RVs). An RV might be multidimensional though -- see num_dims.
    num_dims: number of dimensions per RV. E.g. you can model an 32x32 RGB image as an 32x32 array of three dimensional
              RVs.
    num_input_distributions: number of distributions per input region (K in the paper).
    num_sums: number of sum nodes per internal region (K in the paper).
    num_classes: number of outputs of the PC.
    exponential_family: which exponential family to use; (sub-class ExponentialFamilyTensor).
    exponential_family_args: arguments for the exponential family, e.g. trial-number N for Binomial.
    use_em: determines if the internal em algorithm shall be used; otherwise you might use e.g. SGD.
    online_em_frequency: how often shall online be triggered in terms, of batches? 1 means after each batch, None means
                         batch EM. In the latter case, EM updates must be triggered manually after each epoch.
    online_em_stepsize: stepsize for inline EM. Only relevant if online_em_frequency not is None.
    """

    def __init__(self,
                 rg_structure: str,
                 layer_type,
                 num_var,
                 num_sums,
                 num_input,
                 exponential_family,
                 exponential_family_args,
                 r=1,
                 prod_exp: bool = False,
                 pd_num_pieces: int = None,
                 shrink: bool = False):

        self.r = r
        self.num_var = num_var
        self.num_dims = 1
        self.num_input_distributions = num_input
        self.num_sums = num_sums
        self.num_classes = 1
        self.exponential_family = exponential_family
        self.exponential_family_args = exponential_family_args
        self.use_em = False
        self.online_em_frequency = 1  # unused ?
        self.online_em_stepsize = 0.05  # unused ?
        self.prod_exp = prod_exp
        self.rg_structure = rg_structure
        self.pd_num_pieces = pd_num_pieces
        self.shrink: bool = shrink

        self.layer_type = layer_type


class LoRaEinNetwork(torch.nn.Module):

    def __init__(self, graph, args):
        """Make an EinsumNetwork."""
        super(LoRaEinNetwork, self).__init__()

        check_flag, check_msg = Graph.check_graph(graph)
        if not check_flag:
            raise AssertionError(check_msg)
        self.graph = graph

        self.args = args

        if len(Graph.get_roots(self.graph)) != 1:
            raise AssertionError("Currently only EinNets with single root node supported.")

        root = Graph.get_roots(self.graph)[0]
        if tuple(range(self.args.num_var)) != root.scope:
            raise AssertionError("The graph should be over tuple(range(num_var)).")

        for node in Graph.get_leaves(self.graph):
            node.num_dist = self.args.num_input_distributions

        for node in Graph.get_sums(self.graph):
            if node is root:
                node.num_dist = self.args.num_classes
            else:
                node.num_dist = self.args.num_sums

        # Algorithm 1 in the paper -- organize the PC in layers  NOT BOTTOM UP !!!
        # self.graph_layers = Graph.topological_layers_bottom_up(self.graph)
        self.graph_layers = Graph.topological_layers(self.graph)


        # input layer
        einet_layers = [FactorizedLeafLayer(self.graph_layers[0],
                                            self.args.num_var,
                                            self.args.num_dims,
                                            self.args.exponential_family,
                                            self.args.exponential_family_args,
                                            use_em=False)]  # note: enforcing this todo: restore

        if not self.args.shrink:
            def k_gen():
                while True:
                    print(self.args.num_sums)
                    yield self.args.num_sums
        else:
            def k_gen():
                k_list = list(reversed([2**i for i in range(5, int(np.log2(self.args.num_sums)))]))
                # k_list = sorted(k_list + k_list)

                while True:
                    if len(k_list) > 1:
                        yield k_list.pop(0)
                    else:
                        yield k_list[0]

        k = k_gen()

        # internal layers
        for c, layer in enumerate(self.graph_layers[1:]):
            if c % 2 == 0:  # product layer, that's a partition layer in the graph

                num_sums = set([n.num_dist for p in layer for n in graph.pred[p]])
                if len(num_sums) != 1:
                    raise AssertionError(f"For internal {c} there are {len(num_sums)} nums sums")
                num_sums = list(num_sums)[0]

                layer_type: str = str.lower(self.args.layer_type)
                if layer_type == "hcpt":
                    einet_layers.append(HCPTEinsumLayer(self.graph, layer, einet_layers,
                                                        prod_exp=self.args.prod_exp, k=next(k)))
                elif layer_type == "hcpt-shared":
                    einet_layers.append(HCPTSharedEinsumLayer(self.graph, layer, einet_layers,
                                                              prod_exp=self.args.prod_exp, k=next(k)))
                elif layer_type == "cp":
                    if num_sums > 1:
                        einet_layers.append(CPEinsumLayer(self.graph, layer, einet_layers,
                                                          r=self.args.r,
                                                          prod_exp=self.args.prod_exp, k=next(k)))
                    else:
                        einet_layers.append(RescalEinsumLayer(self.graph, layer, einet_layers, k=next(k)))
                elif layer_type == "cp-shared":
                    if num_sums > 1:
                        einet_layers.append(CPSharedEinsumLayer(self.graph, layer, einet_layers,
                                                                r=self.args.r,
                                                                prod_exp=self.args.prod_exp, k=next(k)))
                    else:
                        einet_layers.append(RescalEinsumLayer(self.graph, layer, einet_layers, k=next(k)))
                elif layer_type == "rescal":
                    if self.args.prod_exp:
                        warnings.warn("Rescal has numerical properties of prod_exp False")
                    einet_layers.append(RescalEinsumLayer(self.graph, layer, einet_layers, k=next(k)))
                elif layer_type == "hcpt-lolo":
                    if num_sums > 1:
                        einet_layers.append(HCPTLoLoEinsumLayer(self.graph, layer,
                                                                einet_layers,
                                                                r=self.args.r,
                                                                prod_exp=self.args.prod_exp, k=next(k)))
                    else:
                        einet_layers.append(HCPTEinsumLayer(self.graph, layer, einet_layers,
                                                            prod_exp=self.args.prod_exp, k=next(k)))

                elif layer_type == "hcpt-lolo-shared":
                    if num_sums > 1:
                        einet_layers.append(HCPTLoLoSharedEinsumLayer(self.graph, layer,
                                                                      einet_layers,
                                                                      r=self.args.r,
                                                                      prod_exp=self.args.prod_exp,
                                                                      k=next(k)))
                    else:
                        einet_layers.append(HCPTEinsumLayer(self.graph, layer, einet_layers,
                                                            prod_exp=self.args.prod_exp, k=next(k)))
                else:
                    raise AssertionError("Unknown layer type")

            else:  # sum layer, that's a region layer in the graph
                # the Mixing layer is only for regions which have multiple partitions as children.
                multi_sums = [n for n in layer if len(graph.succ[n]) > 1]
                if multi_sums:
                    einet_layers.append(EinsumMixingLayer(graph, multi_sums, einet_layers[-1]))

        self.einet_layers = torch.nn.ModuleList(einet_layers)
        self.exp_reparam: bool = None
        self.mixing_softmax: bool = None

    def get_device(self):
        return self.einet_layers[-1].params.device

    def initialize(self, init_dict=None, exp_reparam=False, mixing_softmax=False):
        self.exp_reparam = exp_reparam
        assert not mixing_softmax
        assert not exp_reparam
        """
        Initialize layers.

        :param init_dict: None; or
                          dictionary int->initializer; mapping layer index to initializers; or
                          dictionary layer->initializer;
                          the init_dict does not need to have an initializer for all layers
        :return: None
        """
        if init_dict is None:
            init_dict = dict()
        if all([type(k) == int for k in init_dict.keys()]):
            init_dict = {self.einet_layers[k]: init_dict[k] for k in init_dict.keys()}
        for layer in self.einet_layers:
            layer.initialize(init_dict.get(layer, 'default'))

    def set_marginalization_idx(self, idx):
        """Set indices of marginalized variables."""
        self.einet_layers[0].set_marginalization_idx(idx)

    def get_marginalization_idx(self):
        """Get indices of marginalized variables."""
        return self.einet_layers[0].get_marginalization_idx()

    def partition_function(self, x=None):
        old_marg_idx = self.get_marginalization_idx()
        self.set_marginalization_idx(np.array(range(self.args.num_var)))

        if x is None:  # TODO: check this, size=(1, self.args.num_var, self.args.num_dims) should be appropriate
            fake_data = torch.ones(size=(1, self.args.num_var), device=self.einet_layers[-1].params.device)
            z = self.forward(fake_data)
        else:
            z = self.forward(x)

        self.set_marginalization_idx(old_marg_idx)
        return z

    def forward(self, x, plot_dict=None):
        """
        Evaluate the EinsumNetwork feed forward.
        :param x: the shape is (batch_size, self.num_var, self.num_dims)
        """

        input_layer = self.einet_layers[0]
        input_layer(x=x)

        if plot_dict is not None:
            if "0" not in plot_dict.keys():
                plot_dict["0"] = torch.sum(input_layer.prob, 0)
            else:
                plot_dict["0"] = plot_dict["0"] + torch.sum(input_layer.prob, 0)

        for num_layer, einsum_layer in enumerate(self.einet_layers[1:]):
            einsum_layer()

            if plot_dict is not None:
                if str(num_layer+1) not in plot_dict.keys():
                    plot_dict[str(num_layer+1)] = torch.sum(einsum_layer.prob, 0)
                else:
                    plot_dict[str(num_layer + 1)] = plot_dict[str(num_layer+1)] + torch.sum(einsum_layer.prob, 0)
        return self.einet_layers[-1].prob[:, :, 0]

    def get_layers(self):
        return self.einet_layers

    def forward_layer(self, layer, x=None):

        if layer is self.einet_layers[0]:
            assert x is not None
            layer(x=x)
        else:
            layer()

    def backtrack(self, num_samples=1, class_idx=0, x=None, mode='sampling', **kwargs):
        """
        Perform backtracking; for sampling or MPE approximation.
        """
        raise NotImplementedError

        # dicts {layer, list}
        sample_idx = {l: [] for l in self.einet_layers}
        dist_idx = {l: [] for l in self.einet_layers}
        reg_idx = {l: [] for l in self.einet_layers}

        root = self.einet_layers[-1]

        if x is not None:
            self.forward(x)
            num_samples = x.shape[0]

        # [0, ..., num_samples - 1]
        sample_idx[root] = list(range(num_samples))
        dist_idx[root] = [class_idx] * num_samples  # [0, 0, ..., 0]
        reg_idx[root] = [0] * num_samples  # [0, 0, ..., 0]

        for layer in reversed(self.einet_layers):

            if not sample_idx[layer]:
                continue

            if type(layer) == EinsumLayer:

                ret = layer.backtrack(dist_idx[layer],
                                      reg_idx[layer],
                                      sample_idx[layer],
                                      use_evidence=(x is not None),
                                      mode=mode,
                                      **kwargs)
                dist_idx_left, dist_idx_right, reg_idx_left, reg_idx_right, layers_left, layers_right = ret

                for c, layer_left in enumerate(layers_left):
                    sample_idx[layer_left].append(sample_idx[layer][c])
                    dist_idx[layer_left].append(dist_idx_left[c])
                    reg_idx[layer_left].append(reg_idx_left[c])

                for c, layer_right in enumerate(layers_right):
                    sample_idx[layer_right].append(sample_idx[layer][c])
                    dist_idx[layer_right].append(dist_idx_right[c])
                    reg_idx[layer_right].append(reg_idx_right[c])

            elif type(layer) == EinsumMixingLayer:

                ret = layer.backtrack(dist_idx[layer],
                                      reg_idx[layer],
                                      sample_idx[layer],
                                      use_evidence=(x is not None),
                                      mode=mode,
                                      **kwargs)
                dist_idx_out, reg_idx_out, layers_out = ret

                for c, layer_out in enumerate(layers_out):
                    sample_idx[layer_out].append(sample_idx[layer][c])
                    dist_idx[layer_out].append(dist_idx_out[c])
                    reg_idx[layer_out].append(reg_idx_out[c])

            elif type(layer) == FactorizedLeafLayer:

                unique_sample_idx = sorted(list(set(sample_idx[layer])))
                if unique_sample_idx != sample_idx[root]:
                    raise AssertionError("This should not happen.")

                dist_idx_sample = []
                reg_idx_sample = []
                for sidx in unique_sample_idx:
                    dist_idx_sample.append([dist_idx[layer][c] for c, i in enumerate(sample_idx[layer]) if i == sidx])
                    reg_idx_sample.append([reg_idx[layer][c] for c, i in enumerate(sample_idx[layer]) if i == sidx])

                samples = layer.backtrack(dist_idx_sample, reg_idx_sample, mode=mode, **kwargs)

                if self.args.num_dims == 1:
                    samples = torch.squeeze(samples, 2)

                if x is not None:
                    marg_idx = layer.get_marginalization_idx()
                    keep_idx = [i for i in range(self.args.num_var) if i not in marg_idx]
                    samples[:, keep_idx] = x[:, keep_idx]

                return samples

    def sample(self, num_samples=1, class_idx=0, x=None, **kwargs):
        return self.backtrack(num_samples=num_samples, class_idx=class_idx, x=x, mode='sample', **kwargs)

    def mpe(self, num_samples=1, class_idx=0, x=None, **kwargs):
        return self.backtrack(num_samples=num_samples, class_idx=class_idx, x=x, mode='argmax', **kwargs)


    """
    def em_set_hyperparams(self, online_em_frequency, online_em_stepsize, purge=True):
        for l in self.einet_layers:
            l.em_set_hyperparams(online_em_frequency, online_em_stepsize, purge)

    def em_process_batch(self):
        for l in self.einet_layers:
            l.em_process_batch()

    def em_update(self):
        for l in self.einet_layers:
            l.em_update()
    """


def check_network_parameters(einet: LoRaEinNetwork):

    for n, layer in enumerate(einet.einet_layers):
        if type(layer) == FactorizedLeafLayer:
            continue
        else:
            clamp_value = layer.clamp_value
            for par in layer.parameters():
                if torch.isinf(par).any():
                    raise AssertionError(f"Inf parameter at {n}, {type(layer)}")
                if torch.isnan(par).any():
                    raise AssertionError(f"NaN parameter at {n}, {type(layer)}")
                if not torch.all(par >= clamp_value):
                    raise AssertionError(f"Parameter less than clamp value at {n}, {type(layer)}")

