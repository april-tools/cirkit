from typing import Any, Optional, cast

import torch
from torch import Tensor, nn

from cirkit.layers.sum_product import SumProductLayer
from cirkit.utils.reparams import ReparamFunction, reparam_id


class CPCollapsedLayer(SumProductLayer):
    """Candecomp Parafac (decomposition) layer, collapsing the C matrix."""

    # TODO: better way to call init by base class?
    # TODO: better default value
    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        num_folds: int = 1,
        fold_mask: Optional[torch.Tensor] = None,
        *,
        reparam: ReparamFunction = reparam_id,
        **_: Any,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int): The arity of the product units.
            num_folds (int): The number of folds.
            fold_mask (Optional[torch.Tensor]): The mask to apply to the folded parameter tensors.
            reparam: The reparameterization function.
            prod_exp (bool): Whether to compute products in linear space rather than in log-space.
        """
        super().__init__(
            num_input_units, num_output_units, num_folds=num_folds, fold_mask=fold_mask
        )
        assert arity > 0
        self.arity = arity
        self.reparam = reparam

        self.params = nn.Parameter(
            torch.empty(self.num_folds, arity, num_input_units, num_output_units)
        )

        # TODO: get torch.default_float_dtype
        # (float ** float) is not guaranteed to be float, but here we know it is
        self.param_clamp_value["min"] = cast(
            float,
            torch.finfo(self.params.dtype).smallest_normal ** 0.5,
        )

        self.reset_parameters()

    # TODO: use bmm to replace einsum?
    def _forward(self, x: Tensor) -> Tensor:
        if self.fold_mask is not None:  # pylint: disable=consider-ternary-expression
            fold_mask = self.fold_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
        else:
            fold_mask = None
        weight = self.reparam(self.params, fold_mask)  # (F, H, K, J)
        return torch.einsum("fhkj,fhkb->fhjb", weight, x)  # (F, H, J, B)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the main Einsum operation of the layer.

        :param inputs: value in log space for left child.
        :return: result of the left operations, in log-space.
        """
        m: Tensor = torch.max(inputs, dim=2, keepdim=True)[0]  # (F, H, 1, B)
        x = torch.exp(inputs - m)  # (F, H, K, B)
        x = self._forward(x)  # (F, H, J, B)
        x = torch.log(x)
        if self.fold_mask is not None:
            x = torch.nan_to_num(x, neginf=0)
        return torch.sum(x + m, dim=1)  # (F, J, B)
