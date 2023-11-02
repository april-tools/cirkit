from typing import Any, Optional, cast

import torch
from torch import Tensor, nn

from cirkit.layers.sum_product.sum_product import SumProductLayer
from cirkit.utils.reparams import ReparamFunction, reparam_id


def _cp_einsum(x: Tensor, w: Tensor) -> Tensor:
    # x: (F, H, K, B)
    # w: (F, H, K, J)
    # output: (F, H, J, B)
    return torch.einsum("fhkj,fhkb->fhjb", w, x)


def _cp_uncollapsed_einsum(x: Tensor, w: Tensor) -> Tensor:
    # x: (F, K, B)
    # w: (F, K, J)
    # output: (F, J, B)
    return torch.einsum("fkj,fkb->fjb", w, x)


def _cp_shared_einsum(x: Tensor, w: Tensor) -> Tensor:
    # x: (F, H, K, B)
    # w: (H, K, J)
    # output: (F, H, J, B)
    return torch.einsum("hkj,fhkb->fhjb", w, x)


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
        reparam: ReparamFunction = reparam_id,
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
        self.reparam = reparam
        self.uncollapsed = uncollapsed

        if uncollapsed:
            assert rank > 0
            self.params_in = nn.Parameter(torch.empty(self.num_folds, arity, num_input_units, rank))
            self.params_out = nn.Parameter(torch.empty(self.num_folds, rank, num_output_units))
        else:
            self.params_in = nn.Parameter(
                torch.empty(self.num_folds, arity, num_input_units, num_output_units)
            )

        # TODO: get torch.default_float_dtype
        # (float ** float) is not guaranteed to be float, but here we know it is
        self.param_clamp_value["min"] = cast(
            float,
            torch.finfo(self.params_in.dtype).smallest_normal ** 0.5,
        )

        self.reset_parameters()

    def _reparam_in(self) -> Tensor:
        return self.reparam(self.params_in, None)  # (F, H, K, J)

    def _reparam_out(self) -> Tensor:
        params_out = self.reparam(self.params_out, None)  # (F, K, J)
        return params_out

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the main Einsum operation of the layer.

        :param inputs: value in log space for left child.
        :return: result of the left operations, in log-space.
        """
        params_in = self._reparam_in()
        # TODO: recover the log_trick here
        m: Tensor = torch.max(inputs, dim=2, keepdim=True)[0]  # (F, H, 1, B)
        x = torch.exp(inputs - m)  # (F, H, K, B)
        x = _cp_einsum(x, params_in)  # (F, H, J, B)
        x = torch.log(x) + m
        if self.fold_mask is not None:
            x = x * self.fold_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = torch.sum(x, dim=1)  # (F, J, B)
        if not self.uncollapsed:
            return x
        params_out = self._reparam_out()
        m: Tensor = torch.max(x, dim=1, keepdim=True)[0]  # type: ignore[no-redef]
        x = torch.exp(x - m)  # (F, R, B)
        x = _cp_uncollapsed_einsum(x, params_out)  # (F, K, B)
        x = torch.log(x) + m
        return x


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
        reparam: ReparamFunction = reparam_id,
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
        reparam: ReparamFunction = reparam_id,
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

        self.params = nn.Parameter(torch.empty(arity, num_input_units, num_output_units))

        # TODO: get torch.default_float_dtype
        # (float ** float) is not guaranteed to be float, but here we know it is
        self.param_clamp_value["min"] = cast(
            float,
            torch.finfo(self.params.dtype).smallest_normal ** (1 / 2),
        )

        self.reset_parameters()

    def _reparam(self) -> Tensor:
        return self.reparam(self.params, None)  # (H, K, J)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the main Einsum operation of the layer.

        :param inputs: value in log space for left child.
        :return: result of the left operations, in log-space.
        """
        params = self._reparam()
        m: Tensor = torch.max(inputs, dim=2, keepdim=True)[0]  # (F, H, 1, B)
        x = torch.exp(inputs - m)  # (F, H, K, B)
        if len(params.shape) == 3:  # pylint: disable=consider-ternary-expression
            x = _cp_shared_einsum(x, params)  # (F, H, K, B)
        else:
            x = _cp_einsum(x, params)  # (F, H, K, B)
        x = torch.log(x) + m
        if self.fold_mask is not None:
            x = x * self.fold_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return torch.sum(x, dim=1)  # (F, K, B)
