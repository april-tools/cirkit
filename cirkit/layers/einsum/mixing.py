from typing import List

import torch
from torch import Tensor, nn

from cirkit.layers.einsum import EinsumLayer
from cirkit.layers.layer import Layer
from cirkit.region_graph import RegionNode

# TODO: rework docstrings


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


class EinsumMixingLayer(Layer):
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

    # TODO: might be good to doc params and buffers here
    # to be registered as buffer
    params_mask: Tensor

    def __init__(self, region_layer: List[RegionNode], einsum_layer: EinsumLayer):
        """Init class.

        :param region_layer: the nodes of the current layer (see constructor of \
            EinsumNetwork), which have multiple children
        :param einsum_layer:
        """
        super().__init__()

        self.region_layer = region_layer

        k = set(region.k for region in region_layer)
        assert len(k) == 1, f"The K of region nodes in the same layer must be the same, got {k}."
        self.k = k.pop()

        self.max_components = max(len(region.inputs) for region in region_layer)

        # einsum is actually the only layer which gives input to EinsumMixingLayer
        # we keep it in a list, since otherwise it gets registered as a torch sub-module
        self.input_layer_as_list = [einsum_layer]

        param_shape = (self.k, len(region_layer), self.max_components)
        # param_shape = (len(self.nodes), self.max_components) for better perf
        # TODO: test best perf?

        self.param = nn.Parameter(torch.empty(param_shape))
        # TODO: what's the use of params_mask?
        self.register_buffer("params_mask", torch.ones_like(self.param))

        self.param_clamp_value["min"] = torch.finfo(self.param.dtype).smallest_normal

        # self.reset_parameters()  # TODO: params_mask caused a mess

    def reset_parameters(self) -> None:
        """Reset parameters to default initialization: U(0.01, 0.99) with normalization."""
        nn.init.uniform_(self.param, 0.01, 0.99)

        with torch.no_grad():
            if self.params_mask is not None:
                # TODO: assume mypy bug with __mul__ and __div__
                self.param *= self.params_mask  # type: ignore[misc]

            self.param /= self.param.sum(dim=2, keepdim=True)  # type: ignore[misc]

    # TODO: make forward return something
    # pylint: disable=arguments-differ
    def forward(self, log_input_prob: Tensor) -> Tensor:  # type: ignore[override]
        """Do the forward.

        Args:
            log_input_prob (Tensor): The input.

        Returns:
            Tensor: the output.
        """
        # TODO: still the same torch max problem by mypy
        input_max: Tensor = torch.max(log_input_prob, dim=3, keepdim=True)[0]
        input_prob = torch.exp(log_input_prob - input_max)

        # TODO: use a mul or gather?
        assert (self.param * self.params_mask == self.param).all()

        prob = torch.einsum("bonc,onc->bon", input_prob, self.param)
        log_prob = torch.log(prob) + input_max[..., 0]

        self.prob = log_prob
        return self.prob

    # # TODO: how is this useful?
    # # TODO: not refactored
    # def _backtrack(  # type: ignore[misc]
    #     self,
    #     dist_idx: List[int],
    #     node_idx: List[int],
    #     sample_idx: List[int],
    #     params: Tensor,
    #     use_evidence: bool = False,
    #     mode: Literal["sample", "argmax"] = "sample",
    #     **_: Any,
    # ) -> Tuple[List[int], List[int], List[EinsumLayer]]:
    #     """Is helper routine for backtracking in EiNets."""
    #     # TODO: why not at module level
    #     with torch.no_grad():
    #         if use_evidence:
    #             log_prior = torch.log(params[dist_idx, node_idx, :])
    #             log_posterior = log_prior + self.log_input_prob[sample_idx, dist_idx, node_idx, :]
    #             # TODO: torch has logsumexp!!!
    #             posterior = torch.exp(
    #                 log_posterior - torch.logsumexp(log_posterior, 1, keepdim=True)
    #             )
    #         else:
    #             posterior = params[dist_idx, node_idx, :]

    #         # TODO: not pass mode as str, but a callable for selecting idx
    #         if mode == "sample":
    #             idx = _sample_matrix_categorical(posterior)
    #         elif mode == "argmax":
    #             idx = torch.argmax(posterior, -1)
    #         node_idx_out = [
    #             self.mixing_component_idx[self.region_layer[i]][idx[c]]
    #             for c, i in enumerate(node_idx)
    #         ]
    #         # TODO: make sure it's wanted. it's all copies of the same object
    #         layers_out = [self.input_layer_as_list[0]] * len(node_idx)

    #     return dist_idx, node_idx_out, layers_out
