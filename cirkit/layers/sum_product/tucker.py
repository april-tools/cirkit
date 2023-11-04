from typing import Any, Optional, cast

import torch
from torch import Tensor

from cirkit.layers.sum_product.sum_product import SumProductLayer
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils.log_trick import log_func_exp
from cirkit.utils.type_aliases import ReparamFactory

# TODO: rework docstrings


class TuckerLayer(SumProductLayer):
    """Tucker (2) layer."""

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
        reparam: ReparamFactory = ReparamIdentity,
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
        if arity != 2 or fold_mask is not None:
            raise NotImplementedError("Tucker layers can only compute binary product units")

        self.params = reparam(
            (self.num_folds, num_input_units, num_input_units, num_output_units), dim=(1, 2)
        )

        # TODO: get torch.default_float_dtype
        # (float ** float) is not guaranteed to be float, but here we know it is
        self.param_clamp_value["min"] = cast(
            float,
            torch.finfo(self.params.dtype).smallest_normal ** 0.5,
        )

        self.reset_parameters()

    def _forward_linear(self, left: Tensor, right: Tensor) -> Tensor:
        return torch.einsum("pib,pjb,pijo->pob", left, right, self.params())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute the main Einsum operation of the layer.

        :param inputs: value in log space for left child.
        :return: result of the left operations, in log-space.
        """
        log_left, log_right = inputs[:, 0], inputs[:, 1]
        return log_func_exp(log_left, log_right, func=self._forward_linear, dim=1, keepdim=True)
