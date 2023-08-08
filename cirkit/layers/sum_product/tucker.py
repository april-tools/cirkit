import warnings
from typing import Any, Optional, cast

import torch
from torch import Tensor, nn

from cirkit.layers.sum_product import SumProductLayer
from cirkit.utils import log_func_exp
from cirkit.utils.reparams import ReparamFunction, reparam_id

# TODO: rework docstrings


class TuckerLayer(SumProductLayer):
    """Tucker (2) layer."""

    # TODO: better way to call init by base class?
    # TODO: better default value
    def __init__(  # type: ignore[misc]
        self,
        num_input_units: int,
        num_output_units: int,
        num_folds: int = 1,
        fold_mask: Optional[torch.Tensor] = None,
        *,
        reparam: ReparamFunction = reparam_id,
        prod_exp: bool,
        **_: Any,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            num_folds (int): The number of folds.
            fold_mask (Optional[torch.Tensor]): The mask to apply to the folded parameter tensors.
            reparam: The reparameterization function.
            prod_exp (bool): Whether to compute products in linear space rather than in log-space.
        """
        # TODO: for now we don't care about the case of prod_exp False
        if prod_exp:
            warnings.warn("Prod exp not available for Tucker")

        super().__init__(
            num_input_units, num_output_units, num_folds=num_folds, fold_mask=fold_mask
        )
        self.reparam = reparam
        self.prod_exp = prod_exp

        self.params = nn.Parameter(
            torch.empty(self.num_folds, num_input_units, num_input_units, num_output_units)
        )

        # TODO: get torch.default_float_dtype
        # (float ** float) is not guaranteed to be float, but here we know it is
        self.param_clamp_value["min"] = cast(
            float,
            torch.finfo(self.params.dtype).smallest_normal ** (1 / 2),
        )

        self.reset_parameters()

    def _forward_linear(self, left: Tensor, right: Tensor) -> Tensor:
        weight = self.reparam(self.params, self.fold_mask)
        return torch.einsum("pib,pjb,pijo->pob", left, right, weight)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the main Einsum operation of the layer.

        :param inputs: value in log space for left child.
        :return: result of the left operations, in log-space.
        """
        log_left, log_right = inputs[:, 0], inputs[:, 1]

        return log_func_exp(log_left, log_right, func=self._forward_linear, dim=1, keepdim=True)
