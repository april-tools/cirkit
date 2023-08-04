from typing import Callable, List

import torch
from torch import Tensor, nn

from cirkit.layers.layer import Layer
from cirkit.region_graph import RegionNode
from cirkit.utils import log_func_exp
from cirkit.utils.reparams import reparam_id

# TODO: rework docstrings


class MixingLayer(Layer):
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
    SumProductLayer, yielding a log-density tensor of shape
        (batch_size, vector_length, num_nodes).
    In this example num_nodes is 5, since the are 5 product nodes (or 5 singleton \
        sum nodes). The MixingLayer
    then simply mixes sums from the first layer, to yield 2 sums. This is just an \
        over-parametrization of the original
    excerpt.
    """

    # TODO: num_output_units is num_input_units
    def __init__(
        self,
        rg_nodes: List[RegionNode],
        num_output_units: int,
        max_components: int,
        *,
        reparam: Callable[[torch.Tensor], torch.Tensor] = reparam_id,
    ) -> None:
        """Init class.

        Args:
            rg_nodes (List[PartitionNode]): The region graph's partition node of the layer.
            num_output_units (int): The number of output units.
            max_components (int): Max number of mixing components.
            reparam: The reparameterization function.
        """
        super().__init__()
        self.rg_nodes = rg_nodes
        self.reparam = reparam

        # TODO: what need to be saved to self?
        self.num_output_units = num_output_units

        # TODO: test best perf?
        # param_shape = (len(self.nodes), self.max_components) for better perf
        self.params = nn.Parameter(torch.empty(len(rg_nodes), max_components, num_output_units))

        self.param_clamp_value["min"] = torch.finfo(self.params.dtype).smallest_normal
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters to default initialization: U(0.01, 0.99) with normalization."""
        with torch.no_grad():
            nn.init.uniform_(self.params, 0.01, 0.99)
            self.params /= self.params.sum(dim=1, keepdim=True)  # type: ignore[misc]

    def _forward_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("fck,fckb->fkb", self.reparam(self.params), x)

    # TODO: make forward return something
    # pylint: disable-next=arguments-differ
    def forward(self, log_input: Tensor) -> Tensor:  # type: ignore[override]
        """Do the forward.

        Args:
            log_input (Tensor): The input.

        Returns:
            Tensor: the output.
        """
        return log_func_exp(log_input, func=self._forward_linear, dim=1, keepdim=False)

    # TODO: see commit 084a3685c6c39519e42c24a65d7eb0c1b0a1cab1 for backtrack
