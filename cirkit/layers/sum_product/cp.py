from typing import Any, Optional, cast

import torch
from torch import Tensor, nn

from cirkit.layers.sum_product import SumProductLayer
from cirkit.utils.reparams import ReparamFunction, reparam_id

# TODO: rework docstrings


class CPLayer(SumProductLayer):
    """Candecomp Parafac (decomposition) layer."""

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
        rank: int = 1,
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
            rank (int): The rank of the CP decomposition (i.e., the number of inner units of the \
                layer).
            reparam: The reparameterization function.
        """
        super().__init__(
            num_input_units, num_output_units, num_folds=num_folds, fold_mask=fold_mask
        )
        assert arity > 0
        self.arity = arity
        self.reparam = reparam

        self.params_in = nn.Parameter(torch.empty(self.num_folds, arity, num_input_units, rank))
        self.params_out = nn.Parameter(torch.empty(self.num_folds, rank, num_output_units))

        # TODO: get torch.default_float_dtype
        # (float ** float) is not guaranteed to be float, but here we know it is
        self.param_clamp_value["min"] = cast(
            float, torch.finfo(self.params_in.dtype).smallest_normal ** 0.5
        )
        self.reset_parameters()

    # TODO: use bmm to replace einsum?
    def _forward_in(self, x: Tensor) -> Tensor:
        if self.fold_mask is not None:  # pylint: disable=consider-ternary-expression
            fold_mask = self.fold_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
        else:
            fold_mask = None
        weight = self.reparam(self.params_in, fold_mask)
        return torch.einsum("fhkr,fhkb->fhrb", weight, x)

    def _forward_out(self, x: Tensor) -> Tensor:
        weight = self.reparam(self.params_out, None)
        return torch.einsum("frk,frb->fkb", weight, x)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the main Einsum operation of the layer.

        :param inputs: value in log space for left child.
        :return: result of the left operations, in log-space.
        """
        m: Tensor = torch.max(inputs, dim=2, keepdim=True)[0]  # (F, H, 1, B)
        x = torch.exp(inputs - m)  # (F, H, K, B)
        x = self._forward_in(x)  # (F, H, R, B)
        x = torch.log(x)
        if self.fold_mask is not None:
            x = torch.nan_to_num(x, neginf=0)
        x = torch.sum(x + m, dim=1)  # (F, R, B)
        m: Tensor = torch.max(x, dim=1, keepdim=True)[0]  # type: ignore[no-redef,misc] # (F, 1, B)
        x = torch.exp(x - m)  # (F, R, B)
        x = self._forward_out(x)  # (F, K, B)
        x = torch.log(x) + m
        return x
