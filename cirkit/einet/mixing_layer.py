import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from cirkit.einet.einsum_layer import GenericEinsumLayer
from cirkit.region_graph import RegionGraph, RegionNode

from .sum_layer import SumLayer


@torch.no_grad()
def _sample_matrix_categorical(p: Tensor) -> Tensor:
    """Sample many Categorical distributions represented as rows in a matrix.

    Args:
        p (Tensor): Input.

    Returns:
        Tensor: long tensor.
    """
    cp = torch.cumsum(p[:, :-1], dim=-1)
    rand = torch.rand(cp.shape[0], 1, device=cp.device)
    rand_idx = torch.sum(rand > cp, dim=-1)
    return rand_idx


class EinsumMixingLayer(SumLayer):  # pylint: disable=too-many-instance-attributes
    # TODO: how we fold line here?
    r"""Implement the Mixing Layer, in order to handle sum nodes with multiple children.

    Recall Figure II from above:

           S          S
        /  |  \      / \ 
       P   P  P     P  P
      /\   /\  /\  /\  /\ 
     N  N N N N N N N N N

    Figure II


    We implement such excerpt as in Figure III, splitting sum nodes with multiple \
        children in a chain of two sum nodes:

            S          S
        /   |  \      / \ 
       S    S   S    S  S
       |    |   |    |  |
       P    P   P    P  P
      /\   /\  /\   /\  /\ 
     N  N N N N N N N N N

    Figure III


    The input nodes N have already been computed. The product nodes P and the \
        first sum layer are computed using an
    EinsumLayer, yielding a log-density tensor of shape
        (batch_size, vector_length, num_nodes).
    In this example num_nodes is 5, since the are 5 product nodes (or 5 singleton \
        sum nodes). The EinsumMixingLayer
    then simply mixes sums from the first layer, to yield 2 sums. This is just an \
        over-parametrization of the original
    excerpt.
    """

    # to be registered as buffer
    params_mask: Tensor
    padded_idx: Tensor

    # TODO: generic container to accept?
    def __init__(
        self, graph: RegionGraph, nodes: List[RegionNode], einsum_layer: GenericEinsumLayer
    ):
        """Init class.

        :param graph: the PC graph (see Graph.py)
        :param nodes: the nodes of the current layer (see constructor of \
            EinsumNetwork), which have multiple children
        :param einsum_layer:
        """
        super().__init__()

        self.nodes = nodes

        num_sums = set(n.num_dist for n in nodes)
        assert (
            len(num_sums) == 1
        ), "Number of distributions must be the same for all regions in one layer."
        self.num_sums = num_sums.pop()

        # TODO: directly return a list?
        self.max_components = max(len(list(graph.get_node_input(n))) for n in nodes)

        # einsum_layer is actually the only layer which gives input to EinsumMixingLayer
        # we keep it in a list, since otherwise it gets registered as a torch sub-module
        self.layers = [einsum_layer]
        self.mixing_component_idx = einsum_layer.mixing_component_idx
        assert (
            einsum_layer.dummy_idx is not None
        ), "EinsumLayer has not set a dummy index for padding."

        param_shape = (self.num_sums, len(self.nodes), self.max_components)
        # param_shape = (len(self.nodes), self.max_components) for better perf
        # TODO: test best perf?

        # The following code does some bookkeeping.
        # padded_idx indexes into the log-density tensor of the previous
        # EinsumLayer, padded with a dummy input which
        # outputs constantly 0 (-inf in the log-domain), see class EinsumLayer.
        padded_idx: List[int] = []
        params_mask = torch.ones(param_shape)
        for c, node in enumerate(self.nodes):
            num_components = len(self.mixing_component_idx[node])
            padded_idx += self.mixing_component_idx[node]
            padded_idx += [einsum_layer.dummy_idx] * (self.max_components - num_components)
            if self.max_components > num_components:
                params_mask[:, c, num_components:] = 0.0
            node.einet_address.layer = self
            node.einet_address.idx = c

        # TODO: originally init here? why?
        # super().__init__()

        # TODO: so should put where?
        ####### CODE ORIGINALLY FROM SUMLAYER
        self.params_shape = param_shape
        self.params: nn.Parameter = None  # type: ignore[assignment]
        self.normalization_dims = (2,)
        self.register_buffer("params_mask", params_mask)
        ############## END

        self.register_buffer("padded_idx", torch.tensor(padded_idx))

    def num_of_param(self) -> int:
        """Return the total number of parameters of the layer.

        :return: the total number of parameters of the layer.
        """
        return math.prod(self.params_shape)

    # TODO: why in both children but not base class
    @property
    def clamp_value(self) -> float:
        """Value for parameters clamping to keep all probabilities greater than 0.

        :return: value for parameters clamping
        """
        return torch.finfo(self.params.dtype).smallest_normal

    def clamp_params(self, clamp_all: bool = False) -> None:
        """Clamp parameters such that they are non-negative and is impossible to \
            get zero probabilities.

        This involves using a constant that is specific on the computation.

        Args:
            clamp_all (bool, optional): Whether to clamp all. Defaults to False.
        """
        # TODO: don't use .data but what about grad of nn.Param?
        if clamp_all or self.params.requires_grad:
            self.params.data.clamp_(min=self.clamp_value)

    def initialize(self) -> None:
        """Initialize the parameters for this SumLayer."""
        # TODO: is it good to do so? or use value assign, e.g. copy_?

        # TODO: we should extract this as a shared util
        params = 0.01 + 0.98 * torch.rand(self.params_shape)
        # assert torch.all(params >= 0)

        # TODO: really any grad here?
        with torch.no_grad():
            if self.params_mask is not None:
                params *= self.params_mask

            params /= params.sum(self.normalization_dims, keepdim=True)

        # assert torch.all(params >= 0)
        self.params = torch.nn.Parameter(params)

    # TODO: there's a get_parameter in nn.Module?
    # TODO: why can be a dict
    def get_parameters(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get parameters.

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: The params.
        """
        return self.params

    # TODO: make forward return something. and maybe it's not good to have an extra _forward
    def _forward(self, params: Optional[Tensor] = None) -> None:  # type: ignore[override]
        assert self.layers[0].prob is not None  # TODO: why need this?
        self.child_log_prob = self.layers[0].prob[:, :, self.padded_idx]
        self.child_log_prob = self.child_log_prob.reshape(
            *self.child_log_prob.shape[:2], len(self.nodes), self.max_components
        )

        # TODO: still the same torch max problem by mypy
        max_p: Tensor = torch.max(self.child_log_prob, 3, keepdim=True)[0]
        prob = torch.exp(self.child_log_prob - max_p)

        # TODO: use a mul or gather?
        assert (self.params * self.params_mask == self.params).all()

        output = torch.einsum("bonc,onc->bon", prob, self.params)
        self.prob = torch.log(output) + max_p[:, :, :, 0]

        assert not torch.isnan(self.prob).any()
        assert not torch.isinf(self.prob).any()

    # pylint: disable=too-many-arguments
    def _backtrack(  # type: ignore[misc]
        self,
        dist_idx: List[int],
        node_idx: List[int],
        sample_idx: List[int],
        params: Tensor,
        use_evidence: bool = False,
        mode: Literal["sample", "argmax"] = "sample",
        **_: Any,
    ) -> Tuple[List[int], List[int], List[GenericEinsumLayer]]:
        """Is helper routine for backtracking in EiNets."""
        # TODO: why not at module level
        with torch.no_grad():
            if use_evidence:
                log_prior = torch.log(params[dist_idx, node_idx, :])
                log_posterior = log_prior + self.child_log_prob[sample_idx, dist_idx, node_idx, :]
                posterior = torch.exp(
                    log_posterior - torch.logsumexp(log_posterior, 1, keepdim=True)
                )
            else:
                posterior = params[dist_idx, node_idx, :]

            if mode == "sample":
                idx = _sample_matrix_categorical(posterior)
            elif mode == "argmax":
                idx = torch.argmax(posterior, -1)
            node_idx_out = [
                self.mixing_component_idx[self.nodes[i]][idx[c]] for c, i in enumerate(node_idx)
            ]
            # TODO: make sure it's wanted. it's all copies of the same object
            layers_out = [self.layers[0]] * len(node_idx)

        return dist_idx, node_idx_out, layers_out

    # pylint: disable=missing-param-doc
    def backtrack(self, *_: Any, **__: Any) -> Tensor:  # type: ignore[misc]
        """Do nothing.

        Returns:
            Tensor: Nothing.
        """
        return Tensor()
