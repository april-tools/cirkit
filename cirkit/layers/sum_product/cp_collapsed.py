from typing import Any, List, cast

import torch
from torch import Tensor, nn

from cirkit.layers.sum_product.sum_product import SumProductLayer
from cirkit.region_graph import PartitionNode
from cirkit.utils import log_func_exp


class CPCollapsedLayer(SumProductLayer):
    """Candecomp Parafac (decomposition) layer, collapsing the C matrix."""

    # TODO: better way to call init by base class?
    # TODO: better default value
    def __init__(  # type: ignore[misc]
        self,
        rg_nodes: List[PartitionNode],
        num_input_units: int,
        num_output_units: int,
        *,
        prod_exp: bool,
        **_: Any,
    ) -> None:
        """Init class.

        Args:
            rg_nodes (List[PartitionNode]): The region graph's partition node of the layer.
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            prod_exp (bool): Whether to compute products in linear space rather than in log-space.
        """
        super().__init__(rg_nodes, num_input_units, num_output_units)
        self.prod_exp = prod_exp

        self.params_left = nn.Parameter(
            torch.empty(len(rg_nodes), num_input_units, num_output_units)
        )
        self.params_right = nn.Parameter(
            torch.empty(len(rg_nodes), num_input_units, num_output_units)
        )

        # TODO: get torch.default_float_dtype
        # (float ** float) is not guaranteed to be float, but here we know it is
        self.param_clamp_value["min"] = cast(
            float,
            torch.finfo(self.params_left.dtype).smallest_normal ** (1 / 2),
        )

        self.reset_parameters()

    # TODO: use bmm to replace einsum? also axis order?
    def _forward_left_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("fkr,fkb->frb", self.params_left, x)

    def _forward_right_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("fkr,fkb->frb", self.params_right, x)

    def _forward_linear(self, left: Tensor, right: Tensor) -> Tensor:
        left_hidden = self._forward_left_linear(left)
        right_hidden = self._forward_right_linear(right)
        return left_hidden * right_hidden

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the main Einsum operation of the layer.

        :param inputs: value in log space for left child.
        :return: result of the left operations, in log-space.
        """
        log_left, log_right = inputs[:, 0], inputs[:, 1]

        # TODO: do we split into two impls?
        if self.prod_exp:
            return log_func_exp(log_left, log_right, func=self._forward_linear, dim=1, keepdim=True)

        log_left_hidden = log_func_exp(
            log_left, func=self._forward_left_linear, dim=1, keepdim=True
        )
        log_right_hidden = log_func_exp(
            log_right, func=self._forward_right_linear, dim=1, keepdim=True
        )
        return log_left_hidden + log_right_hidden
