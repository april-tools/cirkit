from typing import List

import torch
from torch import Tensor, nn

from cirkit.layers.layer import Layer
from cirkit.region_graph import PartitionNode, RegionGraph

from .generic_einsum_layer import GenericEinsumLayer

# TODO: rework docstrings


class CPEinsumLayer(GenericEinsumLayer):
    """Candecomp Parafac (decomposition) layer."""

    # TODO: original code changed args order. Not sure what impact
    def __init__(  # pylint: disable=too-many-arguments
        self,
        graph: RegionGraph,
        products: List[PartitionNode],
        layers: List[Layer],
        k: int,
        prod_exp: bool,
        r: int = 1,
    ) -> None:
        """Init class.

        Args:
            graph (RegionGraph): The region graph.
            products (List[PartitionNode]): The current product layer.
            layers (List[Layer]): All the layers currently.
            k (int): I don't know.
            prod_exp (bool): I don't know.
            r (int, optional): The rank? Maybe. Defaults to 1.
        """
        super().__init__(graph, products, layers, prod_exp, k)
        self.cp_a = nn.Parameter(torch.empty(self.num_input_dist, r, len(products)))
        self.cp_b = nn.Parameter(torch.empty(self.num_input_dist, r, len(products)))
        self.cp_c = nn.Parameter(torch.empty(self.num_sums, r, len(products)))

    @property
    def clamp_value(self) -> float:
        """Value for parameters clamping to keep all probabilities greater than 0.

        :return: value for parameters clamping.
        """
        smallest_normal = torch.finfo(self.cp_a.dtype).smallest_normal
        # TODO: seems mypy cannot understand **
        return smallest_normal ** (  # type: ignore[no-any-return,misc]
            1 / 3 if self.prod_exp else 1 / 2
        )

    def central_einsum(self, left_prob: Tensor, right_prob: Tensor) -> Tensor:
        """Compute the main Einsum operation of the layer.

        :param left_prob: value in log space for left child.
        :param right_prob: value in log space for right child.
        :return: result of the left operations, in log-space.
        """
        # TODO: max return type cannot be analysed (but strangely sum is normal)
        left_max: Tensor = torch.max(self.left_child_log_prob, dim=1, keepdim=True)[0]
        left_prob = torch.exp(self.left_child_log_prob - left_max)
        right_max: Tensor = torch.max(self.right_child_log_prob, dim=1, keepdim=True)[0]
        right_prob = torch.exp(self.right_child_log_prob - right_max)

        left_hidden = torch.einsum("bip,irp->brp", left_prob, self.cp_a)
        right_hidden = torch.einsum("bjp,jrp->brp", right_prob, self.cp_b)

        if self.prod_exp:
            # TODO: extract log sum exp as routine?
            hidden = left_hidden * right_hidden
            prob = torch.einsum("brp,orp->bop", hidden, self.cp_c)
            log_prob = torch.log(prob) + left_max + right_max
        else:
            log_left_hidden = torch.log(left_hidden) + left_max
            log_right_hidden = torch.log(right_hidden) + right_max
            log_hidden = log_left_hidden + log_right_hidden

            # TODO: same as above
            hidden_max: Tensor = torch.max(log_hidden, 1, keepdim=True)[0]
            hidden = torch.exp(log_hidden - hidden_max)
            prob = torch.einsum("brp,orp->bop", hidden, self.cp_c)
            log_prob = torch.log(prob) + hidden_max

        return log_prob
