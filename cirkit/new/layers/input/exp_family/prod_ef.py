import math
from typing import Any, Dict, Type
from typing_extensions import Self  # FUTURE: in typing from 3.11

import torch
from torch import Tensor

from cirkit.new.layers.input.exp_family.categorical import CategoricalLayer
from cirkit.new.layers.input.exp_family.exp_family import ExpFamilyLayer
from cirkit.new.layers.input.input import InputLayer
from cirkit.new.reparams import BinaryReparam
from cirkit.new.utils.type_aliases import SymbLayerCfg


class ProdEFLayer(ExpFamilyLayer):
    """The product for Exponential Family distribution layers.

    Exponential Family dist:
        f(x|η) = exp(η · T(x) + log_h(x) - A(η)).

    Product:
        f_1*f_2(x|η_1,η_2) = exp(η · T(x) + log_h(x) - A(η)),
        where:
        - η = concat(η_1, η_2),
        - T(x) = concat(T_1(x), T_2(x)),
        - log_h(x) = log_h_1(x) + log_h_2(x),
        - A(η) = A_1(η_1) + A_2(η_2).

    However the A here is not the log partition anymore, so get_integral should not be 1s.
    """

    # DISABLE: It's designed to have these arguments.
    # IGNORE: Unavoidable for kwargs.
    def __init__(  # type: ignore[misc]  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        reparam: BinaryReparam,
        ef1_cls: Type[ExpFamilyLayer],
        ef1_kwargs: Dict[str, Any],
        ef2_cls: Type[ExpFamilyLayer],
        ef2_kwargs: Dict[str, Any],
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (BinaryReparam): The reparameterization for layer parameters.
            ef1_cls (Type[ExpFamilyLayer]): The class of the first ExpFamilyLayer for product.
            ef1_kwargs (Dict[str,Any]): The kwargs of the first ExpFamilyLayer for product.
            ef2_cls (Type[ExpFamilyLayer]): The class of the second ExpFamilyLayer for product.
            ef2_kwargs (Dict[str,Any]): The kwargs of the second ExpFamilyLayer for product.
        """
        # IGNORE: Unavoidable for kwargs.
        ef1 = ef1_cls(
            num_input_units=num_input_units,
            num_output_units=reparam.reparams[0].shape[1],
            arity=arity,
            reparam=reparam.reparams[0],
            **ef1_kwargs,  # type: ignore[misc]
        )
        ef2 = ef2_cls(
            num_input_units=num_input_units,
            num_output_units=reparam.reparams[1].shape[1],
            arity=arity,
            reparam=reparam.reparams[1],
            **ef2_kwargs,  # type: ignore[misc]
        )
        self.suff_split_point = math.prod(ef1.suff_stats_shape)
        self.suff_stats_shape = (self.suff_split_point + math.prod(ef2.suff_stats_shape),)

        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        # We need suff_stats_shape before __init__, but submodule can only be registered after.
        self.ef1 = ef1
        self.ef2 = ef2

    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, K).

        Returns:
            Tensor: The sufficient statistics T, shape (*B, H, S).
        """
        return torch.cat((self.ef1.sufficient_stats(x), self.ef2.sufficient_stats(x)), dim=-1)

    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, K).

        Returns:
            Tensor: The natural parameters eta, shape (*B, H).
        """
        return self.ef1.log_base_measure(x) + self.ef2.log_base_measure(x)

    def log_partition(self, eta: Tensor) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (H, K, *S).

        Returns:
            Tensor: The log partition function A, shape (H, K).
        """
        # TODO: x.unflatten is not typed
        eta1 = torch.unflatten(
            eta[..., : self.suff_split_point], dim=-1, sizes=self.ef1.suff_stats_shape
        )
        eta2 = torch.unflatten(
            eta[..., self.suff_split_point :], dim=-1, sizes=self.ef2.suff_stats_shape
        )

        return self.ef1.log_partition(eta1) + self.ef2.log_partition(eta2)

    # IGNORE: SymbLayerCfg contains Any.
    @classmethod
    def get_integral(  # type: ignore[misc]
        cls, symb_cfg: SymbLayerCfg[Self]
    ) -> SymbLayerCfg[InputLayer]:
        """Get the symbolic config to construct the definite integral of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.

        Raises:
            NotImplementedError: When "not-yet-implemented feature" is invoked.

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the integral.
        """
        raise NotImplementedError(
            "The integral of ProdEFLayer other than categorical and normal is not yet defined."
        )

    # IGNORE: SymbLayerCfg contains Any.
    @classmethod
    def get_partial(  # type: ignore[misc]
        cls, symb_cfg: SymbLayerCfg[Self], *, order: int = 1, var_idx: int = 0, ch_idx: int = 0
    ) -> SymbLayerCfg[InputLayer]:
        """Get the symbolic config to construct the partial differential w.r.t. the given channel \
        of the given variable in the scope of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.
            order (int, optional): The order of differentiation. Defaults to 1.
            var_idx (int, optional): The variable to diffrentiate. The idx is counted within this \
                layer's scope but not global variable id. Defaults to 0.
            ch_idx (int, optional): The channel of variable to diffrentiate. Defaults to 0.

        Raises:
            TypeError: When this method is called on CategoricalLayer.

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the partial differential w.r.t. the \
                given channel of the given variable.
        """
        # TODO: support nested ProdEFLayer (3 or more products)
        # TODO: how to check continuous/discrete distribution
        # IGNORE: SymbLayerCfg contains Any.
        layer_kwargs = symb_cfg.get("layer_kwargs", {})  # type: ignore[misc]
        if CategoricalLayer in (
            layer_kwargs.get("ef1_cls"),  # type: ignore[misc]
            layer_kwargs.get("ef2_cls"),  # type: ignore[misc]
        ):
            raise TypeError("Cannot differentiate over discrete variables.")

        # TODO: variance issue
        return super().get_partial(
            symb_cfg, order=order, var_idx=var_idx, ch_idx=ch_idx  # type: ignore[misc]
        )
