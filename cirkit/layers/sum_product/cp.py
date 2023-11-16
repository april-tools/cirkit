from typing import Optional, Tuple

import torch
from torch import Tensor

from cirkit.layers.sum_product.sum_product import SumProductLayer
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.reparams.reparam import Reparameterizaion
from cirkit.utils.log_trick import log_func_exp
from cirkit.utils.type_aliases import ReparamFactory


class BaseCPLayer(SumProductLayer):
    """Candecomp Parafac (decomposition) layer."""

    params_in: Optional[Reparameterizaion]
    """The reparameterizaion that gives the parameters for sum units on input, shape as given by \
    dim names, e.g., (F, H, I, O). Can be None to disable this part of computation."""

    params_out: Optional[Reparameterizaion]
    """The reparameterizaion that gives the parameters for sum units on output, shape as given by \
    dim names, e.g., (F, I, O). Can be None to disable this part of computation."""

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
            fold_mask (Optional[Tensor], optional): The mask of valid folds, shape (F, H). \
                Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
            rank (int, optional): The rank of decomposition, i.e., the number of intermediate \
                units. Unused if in collapsed version. Defaults to 0.
            params_in_dim_name (str, optional): The dimension names for the shape of params on \
                input, in einsum notation. Leave empty for no einsum on input. Defaults to "".
            params_out_dim_name (str, optional): The dimension names for the shape of params on \
                output, in einsum notation. Leave empty for no einsum on output. Defaults to "".
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
        assert (
            params_in_dim_name + params_out_dim_name
        ), "CPLayer must have at least one einsum for input or output."
        assert (rank > 0) == (
            "r" in params_in_dim_name + params_out_dim_name
        ), "The rank must be positive if and only if used."
        # Enforce rank==0 when not used, to ensure the user never falsely treat it as used.
        self.rank = rank

        if params_in_dim_name:
            # TODO: convert to tuple currently required to unpack str, but will be changed in a
            #       future version of mypy. see https://github.com/python/mypy/pull/15511
            i, o = tuple(params_in_dim_name[-2:])
            assert i == "i" and o == ("r" if params_out_dim_name else "o")
            self._einsum_in = f"{params_in_dim_name},fh{i}...->fh{o}..."
            # TODO: currently we can only support this. any elegant impl?
            assert params_in_dim_name[:2] == "fh" or fold_mask is None
            # Only params_in can see the folds and need mask.
            self.params_in = reparam(
                self._infer_shape(params_in_dim_name),
                dim=-2,
                mask=fold_mask.view(
                    fold_mask.shape + (1,) * (len(params_in_dim_name) - fold_mask.ndim)
                )
                if fold_mask is not None
                else None,
            )
        else:
            self._einsum_in = ""
            self.params_in = None

        if params_out_dim_name:
            i, o = tuple(params_out_dim_name[-2:])
            assert i == ("r" if params_in_dim_name else "i") and o == "o"
            self._einsum_out = f"{params_out_dim_name},f{i}...->f{o}..."
            self.params_out = reparam(self._infer_shape(params_out_dim_name), dim=-2)
        else:
            self._einsum_out = ""
            self.params_out = None

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
        # shape (F, H, K, *B) -> (F, H, K, *B)
        return torch.einsum(self._einsum_in, self.params_in(), x)

    def _forward_reduce_log(self, x: Tensor) -> Tensor:
        x = (
            x
            if self.fold_mask is None
            else x
            * self.fold_mask.view(self.fold_mask.shape + (1,) * (x.ndim - self.fold_mask.ndim))
        )
        # TODO: is it better to fuse the above mul with the below sum into a matmul/einsum?
        #       Same also appear at other places
        # TODO: double check how we mask things. do we x or params_in?
        return x.sum(dim=1)  # shape (F, H, K, *B) -> (F, K, *B)

    def _forward_out_linear(self, x: Tensor) -> Tensor:
        assert self.params_out is not None and self._einsum_out
        # shape (F, K, *B) -> (F, K, *B)
        return torch.einsum(self._einsum_out, self.params_out(), x)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, K, *B).

        Returns:
            Tensor: The output of this layer, shape (F, K, *B).
        """
        if self.params_in is not None:
            x = log_func_exp(
                x, func=self._forward_in_linear, dim=2, keepdim=True
            )  # shape (F, H, K, *B)
        x = self._forward_reduce_log(x)  # shape (F, K, *B)
        if self.params_out is not None:
            x = log_func_exp(
                x, func=self._forward_out_linear, dim=1, keepdim=True
            )  # shape (F, K, *B)
        return x


class CollapsedCPLayer(BaseCPLayer):
    """Candecomp Parafac (decomposition) layer, with matrix C collapsed."""

    params_in: Reparameterizaion
    """The reparameterizaion that gives the parameters for sum units on input, \
    shape (F, H, I, O)."""

    params_out: None
    """CollapsedCPLayer does not have sum units on output."""

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
            fold_mask (Optional[Tensor], optional): The mask of valid folds, shape (F, H). \
                Defaults to None.
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

    params_in: Reparameterizaion
    """The reparameterizaion that gives the parameters for sum units on input, \
    shape (F, H, I, R)."""

    params_out: Reparameterizaion
    """The reparameterizaion that gives the parameters for sum units on output, shape (F, R, O)."""

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
            fold_mask (Optional[Tensor], optional): The mask of valid folds, shape (F, H). \
                Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
            rank (int, optional): The rank of decomposition, i.e., the number of intermediate \
                units. Unused if in collapsed version. Defaults to 1.
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

    params_in: Reparameterizaion
    """The reparameterizaion that gives the parameters for sum units on input, shape (H, I, O)."""

    params_out: None
    """SharedCPLayer does not have sum units on output."""

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
            fold_mask (Optional[Tensor], optional): The mask of valid folds, unused here. \
                Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
        """
        # SharedCPLayer does not use fold_mask, but it might be provided through the generic
        # interface, and we simply ignore it.
        # TODO: should we allow fold_mask not None? user may falsely think it's used.
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=None,
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

    The default variant is CollapsedCPLayer, but can be selected through the flags.

    Args:
        num_input_units (int): The number of input units.
        num_output_units (int): The number of output units.
        arity (int, optional): The arity of the layer. Defaults to 2.
        num_folds (int, optional): The number of folds. Defaults to 1.
        fold_mask (Optional[Tensor], optional): The mask of valid folds, shape (F, H). \
            Defaults to None.
        reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
        rank (int, optional): The rank of decomposition, i.e., the number of intermediate units. \
            Unused if in collapsed version. Defaults to 0.
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
        # Note that SharedCPLayer does not use fold_mask, and we don't pass it.
        return SharedCPLayer(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=None,
            reparam=reparam,
        )
    raise NotImplementedError("The shared uncollapsed CP is not implemented.")
