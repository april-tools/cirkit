from typing import Optional, Tuple

import torch
from torch import Tensor

from cirkit.layers.sum_product.sum_product import SumProductLayer
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils.log_trick import log_func_exp
from cirkit.utils.type_aliases import ReparamFactory


class BaseCPLayer(SumProductLayer):
    """Candecomp Parafac (decomposition) layer."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        num_folds: int = 1,
        fold_mask: Optional[Tensor] = None,
        reparam: ReparamFactory = ReparamIdentity,
        rank: int = 0,
        params_in_dim_name: str = "",
        params_out_dim_name: str = "",
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
            num_folds (int, optional): The number of folds. Defaults to 1.
            fold_mask (Optional[Tensor], optional): The mask of valid folds. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
            rank (int, optional): The rank for the uncollapsed version. Defaults to 0.
            params_in_dim_name (str, optional): The dimension names for the shape of params on \
                input, in einsum notation. Leave default no einsum on input. Defaults to "".
            params_out_dim_name (str, optional): The dimension names for the shape of params on \
                output, in einsum notation. Leave default no einsum on output. Defaults to "".
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam,
        )
        params_in_dim_name = params_in_dim_name.lower()
        params_out_dim_name = params_out_dim_name.lower()
        assert rank > 0 or "r" not in params_in_dim_name + params_out_dim_name
        self.rank = rank  # unused if "r" not in params_in_dim_name + params_out_dim_name

        assert (
            not params_in_dim_name
            or params_in_dim_name[-2] == "i"
            and params_in_dim_name[-1] in "or"
        )
        self._einsum_in = (
            f"{params_in_dim_name},fhib->fh{params_in_dim_name[-1]}b" if params_in_dim_name else ""
        )
        self.params_in = (  # only params_in can see the folds and need mask
            reparam(self._infer_shape(params_in_dim_name), dim=-2, mask=fold_mask)
            if params_in_dim_name
            else None
        )
        assert (
            not params_out_dim_name
            or params_out_dim_name[-2] in "ir"
            and params_out_dim_name[-1] == "o"
        )
        self._einsum_out = (
            f"{params_out_dim_name},f{params_out_dim_name[-2]}b->fob" if params_out_dim_name else ""
        )
        self.params_out = (
            reparam(self._infer_shape(params_out_dim_name), dim=-2) if params_out_dim_name else None
        )

        self.reset_parameters()

    # TODO: make this more generic, as a until for whole package?
    def _infer_shape(self, dim_names: str) -> Tuple[int, ...]:
        """Infer params shape from the dim names.

        Args:
            dim_names (str): The names of dims, assumed to be lower case.

        Returns:
            Tuple[int, ...]: The shape for parameter.
        """
        mapping = {
            "f": self.num_folds,
            "h": self.arity,
            "r": self.rank,
            "i": self.num_input_units,
            "o": self.num_output_units,
        }
        return tuple(mapping[name] for name in dim_names)

    def _forward_in_linear(self, x: Tensor) -> Tensor:
        assert self.params_in is not None and self._einsum_in
        # TODO: pylint issue
        return torch.einsum(self._einsum_in, self.params_in(), x)  # pylint: disable=not-callable

    def _forward_reduce_log(self, x: Tensor) -> Tensor:
        x = x if self.fold_mask is None else x * self.fold_mask
        return torch.sum(x, dim=1)

    def _forward_out_linear(self, x: Tensor) -> Tensor:
        assert self.params_out is not None and self._einsum_out
        return torch.einsum(self._einsum_out, self.params_out(), x)  # pylint: disable=not-callable

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer.

        Returns:
            Tensor: The output of this layer.
        """
        if self.params_in is not None:
            x = log_func_exp(x, func=self._forward_in_linear, dim=2, keepdim=True)  # (F, H, K, B)
        x = self._forward_reduce_log(x)  # (F, K, B)
        if self.params_out is not None:
            x = log_func_exp(x, func=self._forward_out_linear, dim=1, keepdim=True)  # (F, K, B)
        return x


class CollapsedCPLayer(BaseCPLayer):
    """Candecomp Parafac (decomposition) layer, with matrix C collapsed."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        num_folds: int = 1,
        fold_mask: Optional[Tensor] = None,
        reparam: ReparamFactory = ReparamIdentity,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
            num_folds (int, optional): The number of folds. Defaults to 1.
            fold_mask (Optional[Tensor], optional): The mask of valid folds. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam,
            params_in_dim_name="fhio",
        )


class UncollapsedCPLayer(BaseCPLayer):
    """Candecomp Parafac (decomposition) layer, without collapsing."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        num_folds: int = 1,
        fold_mask: Optional[Tensor] = None,
        reparam: ReparamFactory = ReparamIdentity,
        rank: int = 1,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
            num_folds (int, optional): The number of folds. Defaults to 1.
            fold_mask (Optional[Tensor], optional): The mask of valid folds. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
            rank (int, optional): The rank for the uncollapsed version. Defaults to 0.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam,
            rank=rank,
            params_in_dim_name="fhir",
            params_out_dim_name="fro",
        )


class SharedCPLayer(BaseCPLayer):
    """Candecomp Parafac (decomposition) layer, with parameter sharing and matrix C collapsed."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        num_folds: int = 1,
        fold_mask: Optional[Tensor] = None,
        reparam: ReparamFactory = ReparamIdentity,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
            num_folds (int, optional): The number of folds. Defaults to 1.
            fold_mask (Optional[Tensor], optional): The mask of valid folds. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=None,  # shared CP does not use fold_mask
            reparam=reparam,
            params_in_dim_name="hio",
        )


def CPLayer(  # pylint: disable=invalid-name,too-many-arguments
    *,
    num_input_units: int,
    num_output_units: int,
    arity: int = 2,
    num_folds: int = 1,
    fold_mask: Optional[Tensor] = None,
    reparam: ReparamFactory = ReparamIdentity,
    rank: int = 1,
    collapsed: bool = True,
    shared: bool = False,
) -> BaseCPLayer:
    """Init CPLayer.

    Args:
        num_input_units (int): The number of input units.
        num_output_units (int): The number of output units.
        arity (int, optional): The arity of the layer. Defaults to 2.
        num_folds (int, optional): The number of folds. Defaults to 1.
        fold_mask (Optional[Tensor], optional): The mask of valid folds. Defaults to None.
        reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
        rank (int, optional): The rank for the uncollapsed version. Defaults to 0.
        collapsed (bool, optional): Whether to use collapsed version. Defaults to True.
        shared (bool, optional): Whether to use shared version. Defaults to False.

    Raises:
        NotImplementedError: When given config does not correspond to any implementation.

    Returns:
        BaseCPLayer: The CPLayer as required.
    """
    if not shared and collapsed:
        return CollapsedCPLayer(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam,
        )
    if not shared and not collapsed:
        return UncollapsedCPLayer(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam,
            rank=rank,
        )
    if shared and collapsed:
        return SharedCPLayer(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=None,  # shared CP does not use fold_mask
            reparam=reparam,
        )
    raise NotImplementedError("CP shared uncollapsed not implemented")
