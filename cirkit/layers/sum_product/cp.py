from typing import Any, Optional, cast

import torch
from torch import Tensor

from cirkit.layers.sum_product.sum_product import SumProductLayer
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils.log_trick import log_func_exp
from cirkit.utils.type_aliases import ReparamFactory


class CPLayer(SumProductLayer):
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
        fold_mask: Optional[Tensor] = None,
        *,
        reparam: ReparamFactory = ReparamIdentity,
        uncollapsed: bool = False,
        rank: int = 1,
        **_: Any,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int): The arity of the product units.
            num_folds (int): The number of folds.
            fold_mask (Optional[Tensor]): The mask to apply to the folded parameter tensors.
            reparam: The reparameterization function.
            uncollapsed: Whether to use the "uncollapsed" implementation.
            rank: The rank in case of using the "uncollapsed" implementation.
        """
        super().__init__(
            num_input_units, num_output_units, num_folds=num_folds, fold_mask=fold_mask
        )
        assert arity > 0
        self.arity = arity
        self.uncollapsed = uncollapsed

        if uncollapsed:
            assert rank > 0
            self.params_in = reparam(
                (self.num_folds, arity, num_input_units, rank), dim=2, mask=fold_mask
            )
            self.params_out = reparam((self.num_folds, rank, num_output_units), dim=1)
        else:
            self.params_in = reparam(
                (self.num_folds, arity, num_input_units, num_output_units), dim=2, mask=fold_mask
            )

        # TODO: get torch.default_float_dtype
        # (float ** float) is not guaranteed to be float, but here we know it is
        # TODO: assuming this is not useful anymore?
        self.param_clamp_value["min"] = cast(
            float,
            torch.finfo(self.params_in.dtype).smallest_normal ** 0.5,
        )

        self.reset_parameters()

    def _forward_in_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("fhkr,fhkb->fhrb", self.params_in(), x)

    def _forward_out_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("frk,frb->fkb", self.params_out(), x)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the main Einsum operation of the layer.

        :param inputs: value in log space for left child.
        :return: result of the left operations, in log-space.
        """
        x = log_func_exp(inputs, func=self._forward_in_linear, dim=2, keepdim=True)
        x = torch.sum(x if self.fold_mask is None else x * self.fold_mask, dim=1)  # (F, K/R, B)
        if not self.uncollapsed:
            return x
        return log_func_exp(x, func=self._forward_out_linear, dim=1, keepdim=True)


class UncollapsedCPLayer(CPLayer):
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
        fold_mask: Optional[Tensor] = None,
        *,
        reparam: ReparamFactory = ReparamIdentity,
        rank: int = 1,
        **_: Any,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int): The arity of the product units.
            num_folds (int): The number of folds.
            fold_mask (Optional[Tensor]): The mask to apply to the folded parameter tensors.
            rank (int): The rank of the CP decomposition (i.e., the number of inner units of the \
                layer).
            reparam: The reparameterization function.
        """
        super().__init__(
            num_input_units,
            num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam,
            uncollapsed=True,
            rank=rank,
        )


class SharedCPLayer(SumProductLayer):
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
        fold_mask: Optional[Tensor] = None,
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
            fold_mask (Optional[Tensor]): The mask to apply to the folded parameter tensors.
            reparam: The reparameterization function.
            prod_exp (bool): Whether to compute products in linear space rather than in log-space.
        """
        super().__init__(
            num_input_units, num_output_units, num_folds=num_folds, fold_mask=fold_mask
        )
        assert arity > 0
        self.arity = arity
        self.reparam = reparam

        self.params = reparam((arity, num_input_units, num_output_units), dim=1)

        # TODO: get torch.default_float_dtype
        # (float ** float) is not guaranteed to be float, but here we know it is
        self.param_clamp_value["min"] = cast(
            float,
            torch.finfo(self.params.dtype).smallest_normal ** (1 / 2),
        )

        self.reset_parameters()

    def _forward_in_linear(self, x: Tensor) -> Tensor:
        return torch.einsum("hko,fhkb->fhob", self.params(), x)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the main Einsum operation of the layer.

        :param inputs: value in log space for left child.
        :return: result of the left operations, in log-space.
        """
        x = log_func_exp(inputs, func=self._forward_in_linear, dim=2, keepdim=True)
        return torch.sum(x if self.fold_mask is None else x * self.fold_mask, dim=1)  # (F, K, B)
