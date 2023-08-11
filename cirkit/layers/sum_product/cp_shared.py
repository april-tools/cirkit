from typing import Any, Optional, cast

import torch
from torch import Tensor, nn

from cirkit.layers.sum_product import SumProductLayer
from cirkit.utils.reparams import ReparamFunction, reparam_id


class CPSharedLayer(SumProductLayer):
    """Candecomp Parafac (decomposition) layer with parameter sharing, collapsing the C matrix."""

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
        prod_exp: bool,
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
        self.prod_exp = prod_exp

        self.params = nn.Parameter(torch.empty(arity, num_input_units, num_output_units))

        # TODO: get torch.default_float_dtype
        # (float ** float) is not guaranteed to be float, but here we know it is
        self.param_clamp_value["min"] = cast(
            float,
            torch.finfo(self.params.dtype).smallest_normal ** (1 / 2),
        )

        self.reset_parameters()

    # TODO: use bmm to replace einsum? also axis order?
    def _forward_in(self, x: Tensor) -> Tensor:
        if self.fold_mask is not None:
            fold_mask = self.fold_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)  # (F, H, 1, 1)
            weight = self.reparam(self.params, fold_mask)  # (F, H, K, J)
            return torch.einsum("fhkj,fhkb->fhjb", weight, x)  # (F, H, J, B)
        weight = self.reparam(self.params, None)  # (H, K, J)
        return torch.einsum("hkj,fhkb->fhjb", weight, x)  # (F, H, J, B)

    def _forward(self, x: Tensor) -> Tensor:
        x = self._forward_in(x)  # (F, H, J, B)
        return torch.prod(x, dim=1)  # (F, J, B)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the main Einsum operation of the layer.

        :param inputs: value in log space for left child.
        :return: result of the left operations, in log-space.
        """
        m: torch.Tensor = torch.max(inputs, dim=2, keepdim=True)[0]  # (F, H, 1, B)
        x = torch.exp(inputs - m)  # (F, H, K, B)
        x = self._forward(x)  # (F, H, K, B)
        x = torch.log(x)
        if self.fold_mask is not None:
            x = torch.nan_to_num(x, nan=0.0)
            m = torch.nan_to_num(m, neginf=0.0)
        return torch.sum(x + m, dim=1)  # (F, K, B)
