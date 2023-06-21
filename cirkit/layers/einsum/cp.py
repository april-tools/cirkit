from typing import Any, List, cast

import torch
from torch import Tensor, nn

from cirkit.region_graph import PartitionNode
from cirkit.utils import log_func_exp

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

        self.param_left = nn.Parameter(torch.empty(self.in_k, r, len(partition_layer)))
        self.param_right = nn.Parameter(torch.empty(self.in_k, r, len(partition_layer)))
        self.param_out = nn.Parameter(torch.empty(self.out_k, r, len(partition_layer)))

        # (float ** float) is not guaranteed to be float, but here we know it is
        self.param_clamp_value["min"] = cast(
            float,
            torch.finfo(self.param_left.dtype).smallest_normal
            ** (1 / 3 if self.prod_exp else 1 / 2),
        )

        self.reset_parameters()

    # TODO: use bmm to replace einsum? also axis order?
    def _forward_left_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("bip,irp->brp", x, self.param_left)

    def _forward_right_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("bip,irp->brp", x, self.param_right)

    def _forward_out_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("brp,orp->bop", x, self.param_out)

    def _forward_linear(self, left: Tensor, right: Tensor) -> Tensor:
        left_hidden = self._forward_left_linear(left)
        right_hidden = self._forward_right_linear(right)
        return self._forward_out_linear(left_hidden * right_hidden)

    def forward(self, log_left: Tensor, log_right: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the main Einsum operation of the layer.

        :param log_left: value in log space for left child.
        :param log_right: value in log space for right child.
        :return: result of the left operations, in log-space.
        """
        # TODO: do we split into two impls?
        if self.prod_exp:
            return log_func_exp(log_left, log_right, func=self._forward_linear, dim=1, keepdim=True)

        log_left_hidden = log_func_exp(
            log_left, func=self._forward_left_linear, dim=1, keepdim=True
        )
        log_right_hidden = log_func_exp(
            log_right, func=self._forward_right_linear, dim=1, keepdim=True
        )
        return log_func_exp(
            log_left_hidden + log_right_hidden, func=self._forward_out_linear, dim=1, keepdim=True
        )
