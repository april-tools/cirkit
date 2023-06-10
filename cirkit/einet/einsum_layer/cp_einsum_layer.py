from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from cirkit.einet.layer import Layer
from cirkit.region_graph import PartitionNode, RegionGraph

from .generic_einsum_layer import GenericEinsumLayer


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
        # TODO: init order?
        self.r = r
        super().__init__(graph, products, layers, prod_exp, k)

    def build_params(self) -> Tuple[Dict[str, nn.Parameter], Dict[str, Tuple[int, ...]]]:
        """Create params dict for the layer (the parameters are uninitialized).

        :return: dict {params_name, params}
        """
        # TODO: I don't know what's the meaning of this function w/o actually building the params
        params_dict = {"cp_a": nn.Parameter(), "cp_b": nn.Parameter(), "cp_c": nn.Parameter()}
        shapes_dict = {
            "cp_a": (self.num_input_dist, self.r, len(self.products)),
            "cp_b": (self.num_input_dist, self.r, len(self.products)),
            "cp_c": (self.num_sums, self.r, len(self.products)),
        }
        # TODO: [int, ...] and [int, int, int] not compatible, only by mypy
        return params_dict, shapes_dict  # type: ignore[return-value]

    @property
    def clamp_value(self) -> float:
        """Value for parameters clamping to keep all probabilities greater than 0.

        :return: value for parameters clamping.
        """
        # TODO: not sure what does this mean. why not self.params_dict["cp_a"]
        par_tensor = list(self.params_dict.items())[0][1]
        smallest_normal = torch.finfo(par_tensor.dtype).smallest_normal
        # TODO: seems mypy cannot understand **
        return smallest_normal ** (  # type: ignore[no-any-return,misc]
            1 / 3 if self.prod_exp else 1 / 2
        )

    # pylint: disable-next=too-many-locals
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

        pa = self.params_dict["cp_a"]
        pb = self.params_dict["cp_b"]
        pc = self.params_dict["cp_c"]

        left_hidden = torch.einsum("bip,irp->brp", left_prob, pa)
        right_hidden = torch.einsum("bjp,jrp->brp", right_prob, pb)

        if self.prod_exp:
            # TODO: extract log sum exp as routine?
            hidden = left_hidden * right_hidden
            prob = torch.einsum("brp,orp->bop", hidden, pc)
            log_prob = torch.log(prob) + left_max + right_max
        else:
            log_left_hidden = torch.log(left_hidden) + left_max
            log_right_hidden = torch.log(right_hidden) + right_max
            log_hidden = log_left_hidden + log_right_hidden

            # TODO: same as above
            hidden_max: Tensor = torch.max(log_hidden, 1, keepdim=True)[0]
            hidden = torch.exp(log_hidden - hidden_max)
            prob = torch.einsum("brp,orp->bop", hidden, pc)
            log_prob = torch.log(prob) + hidden_max

        return log_prob
