from typing import Any, List, cast

import torch
from torch import Tensor, nn

from cirkit.region_graph import PartitionNode

from .einsum import EinsumLayer

# TODO: rework docstrings


class CPLayer(EinsumLayer):
    """Candecomp Parafac (decomposition) layer."""

    # TODO: better way to call init by base class?
    def __init__(  # type: ignore[misc]
        self, partition_layer: List[PartitionNode], k: int, *, prod_exp: bool, r: int = 1, **_: Any
    ) -> None:
        """Init class.

        Args:
            partition_layer (List[PartitionNode]): The current product layer.
            k (int): I don't know.
            prod_exp (bool): whether product is in exp-space.
            r (int, optional): The rank? Maybe. Defaults to 1.
        """
        super().__init__(partition_layer, k)
        self.prod_exp = prod_exp

        # TODO: cp_a is not a good name
        self.cp_a = nn.Parameter(torch.empty(self.in_k, r, len(partition_layer)))
        self.cp_b = nn.Parameter(torch.empty(self.in_k, r, len(partition_layer)))
        self.cp_c = nn.Parameter(torch.empty(self.out_k, r, len(partition_layer)))

        # (float ** float) is not guaranteed to be float, but here we know it is
        self.param_clamp_value["min"] = cast(
            float,
            torch.finfo(self.cp_a.dtype).smallest_normal ** (1 / 3 if self.prod_exp else 1 / 2),
        )

        self.reset_parameters()

    # pylint: disable=too-many-locals
    def _forward_einsum(self, log_left_prob: Tensor, log_right_prob: Tensor) -> Tensor:
        """Compute the main Einsum operation of the layer.

        :param log_left_prob: value in log space for left child.
        :param log_right_prob: value in log space for right child.
        :return: result of the left operations, in log-space.
        """
        # TODO: max return type cannot be analysed (but strangely sum is normal)
        left_max: Tensor = torch.max(log_left_prob, dim=1, keepdim=True)[0]
        left_prob = torch.exp(log_left_prob - left_max)
        right_max: Tensor = torch.max(log_right_prob, dim=1, keepdim=True)[0]
        right_prob = torch.exp(log_right_prob - right_max)

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
