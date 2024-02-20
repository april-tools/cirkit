import math
from typing import List, cast
from typing_extensions import Self  # FUTURE: in typing from 3.11

import torch
from torch import Tensor

from cirkit.new.layers.input.exp_family.categorical import CategoricalLayer
from cirkit.new.layers.input.exp_family.exp_family import ExpFamilyLayer
from cirkit.new.layers.input.exp_family.normal import NormalLayer
from cirkit.new.layers.input.input import InputLayer
from cirkit.new.layers.input.param_constant import ParameterizedConstantLayer
from cirkit.new.reparams import EFProductReparam, UnaryReparam
from cirkit.new.utils.type_aliases import SymbCfgFactory, SymbLayerCfg


class ProdEFLayer(ExpFamilyLayer):
    """The product for Exponential Family distribution layers.

    Exponential Family dist:
        p(x|η) = exp(η · T(x) + log_h(x) - A(η)).

    Product:
        p_1*p_2(x|η_1,η_2) = exp(η · T(x) + log_h(x) - A(η)),
    where:
        - η = concat(η_1, η_2),
        - T(x) = concat(T_1(x), T_2(x)),
        - log_h(x) = log_h_1(x) + log_h_2(x),
        - A(η) = A_1(η_1) + A_2(η_2).

    However the A here is not the log partition anymore, so get_integral should not be 1s.
    """

    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        reparam: EFProductReparam,
        ef1_cfg: SymbLayerCfg[ExpFamilyLayer],
        ef2_cfg: SymbLayerCfg[ExpFamilyLayer],
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (EFProductReparam): The reparameterization for layer parameters.
            ef1_cfg (SymbLayerCfg[ExpFamilyLayer]): The config of the left ExpFamilyLayer for \
                product, should include a reference to a concretized SymbL for EF.
            ef2_cfg (SymbLayerCfg[ExpFamilyLayer]): The config of the right ExpFamilyLayer for \
                product, should include a reference to a concretized SymbL for EF.
        """
        assert isinstance(reparam, EFProductReparam), "Must use a EFProductReparam for ProdEFLayer."
        assert (
            ef1 := ef1_cfg.symb_layer.concrete_layer
        ) is not None, (
            "There should be a concrete Layer corresponding to the SymbLayerCfg at this stage."
        )
        assert (
            ef2 := ef2_cfg.symb_layer.concrete_layer
        ) is not None, (
            "There should be a concrete Layer corresponding to the SymbLayerCfg at this stage."
        )

        self.suff_split_point = math.prod(ef1.suff_stats_shape)
        self.suff_stats_shape = (self.suff_split_point + math.prod(ef2.suff_stats_shape),)

        # NOTE: We need a reparam to be provided in SymbLayerCfg and registered for EFLayer.
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
            x (Tensor): The input x, shape (H, *B, Ki).

        Returns:
            Tensor: The sufficient statistics T, shape (H, *B, *S).
        """
        # shape (H, *B, *S_1), (H, *B, *S_2) -> (H, *B, flatten(S_1)+flatten(S_2)).
        return torch.cat(
            (
                self.ef1.sufficient_stats(x).flatten(start_dim=-len(self.ef1.suff_stats_shape)),
                self.ef2.sufficient_stats(x).flatten(start_dim=-len(self.ef2.suff_stats_shape)),
            ),
            dim=-1,
        )

    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, Ki).

        Returns:
            Tensor: The natural parameters eta, shape (H, *B).
        """
        return self.ef1.log_base_measure(x) + self.ef2.log_base_measure(x)

    def log_partition(self, eta: Tensor, *, eta_normed: bool = False) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (H, Ko, *S).
            eta_normed (bool, optional): Ignored. This layer uses a special reparam for eta. \
                Defaults to False.

        Returns:
            Tensor: The log partition function A, shape (H, Ko).
        """
        # TODO: x.unflatten is not typed
        eta1 = torch.unflatten(
            eta[..., : self.suff_split_point], dim=-1, sizes=self.ef1.suff_stats_shape
        )
        eta2 = torch.unflatten(
            eta[..., self.suff_split_point :], dim=-1, sizes=self.ef2.suff_stats_shape
        )

        # TODO: eta_normed?
        return self.ef1.log_partition(eta1) + self.ef2.log_partition(eta2)

    @classmethod
    def get_integral(cls, symb_cfg: SymbLayerCfg[Self]) -> SymbCfgFactory[InputLayer]:
        """Get the symbolic config to construct the definite integral of this layer.

        Generally, we cannot calculate the integral in closed form, but in some special cases we \
        can, as shown below (LogIntegExp is integration w.r.t. x).

        Assume:
            - T_2(x) = T_1(x) (so η_1, η_2 are of the same shape);
            - log_h_2(x) = log_h_2 (constant).

        By definition:
            LogIntegExp(η_1 · T_1(x) + log_h_1(x)) = A_1(η_1).
        That is:
            LogIntegExp((η_1 + η_2) · T_1(x) + log_h_1(x)) = A_1(η_1 + η_2).
        Therefore:
            LogIntegExp(η · T(x) + log_h(x) - A(η)) \
            = LogIntegExp((η_1 + η_2) · T_1(x) + (log_h_1(x) + log_h_2) - A(η)) \
            = A_1(η_1 + η_2) + log_h_2 - A(η).

        The subscripts can be swapped for constant log_h_1, but note that layer product is NOT \
        commutative. For simplicity we further require in practice that p_1 and p_2 are of the \
        same class (and therefore log_h_* are both constant), which also makes it easy to exteed \
        beyond binary product.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.

        Returns:
            SymbCfgFactory[InputLayer]: The symbolic config for the integral.
        """

        def _get_leaf_layers_cfg(
            config: SymbLayerCfg[ExpFamilyLayer],
        ) -> List[SymbLayerCfg[ExpFamilyLayer]]:
            """Fetch all actual input layers configs in the product."""
            if config.layer_cls != ProdEFLayer:
                return [config]
            # IGNORE: Unavoidable for kwargs.
            return _get_leaf_layers_cfg(
                config.layer_kwargs["ef1_cfg"]  # type: ignore[misc]
            ) + _get_leaf_layers_cfg(
                config.layer_kwargs["ef2_cfg"]  # type: ignore[misc]
            )

        layers_cfgs = _get_leaf_layers_cfg(symb_cfg)
        assert all(
            (layer_cfg.layer_cls == layers_cfgs[0].layer_cls) for layer_cfg in layers_cfgs
        ), "Input layers for the product must be the same class."

        assert all(
            issubclass(layer_cfg.layer_cls, (NormalLayer, CategoricalLayer))
            for layer_cfg in layers_cfgs
        ), "Currently only support product partition for Normal or Categorical."

        def _func(eta: Tensor) -> Tensor:
            """Calculate the partition function of the product."""
            assert (
                layer_all := symb_cfg.symb_layer.concrete_layer
            ) is not None, (
                "There should be a concrete Layer corresponding to the SymbLayerCfg at this stage."
            )
            assert (
                layer_1 := layers_cfgs[0].symb_layer.concrete_layer
            ) is not None, (
                "There should be a concrete Layer corresponding to the SymbLayerCfg at this stage."
            )
            pseudo_inputs = torch.empty(layer_1.arity, layer_1.num_input_units)  # Using *B = ().

            # shape (H, Ko, S) -> (H, Ko, len, *S) -> (H, Ko, *S).
            eta_summed = torch.unflatten(
                eta, dim=-1, sizes=(len(layers_cfgs),) + layer_1.suff_stats_shape
            ).sum(dim=2)

            log_part_1 = layer_1.log_partition(eta_summed)

            log_h_2 = layer_all.log_base_measure(pseudo_inputs) - layer_1.log_base_measure(
                pseudo_inputs
            )

            return torch.sum(log_part_1 + log_h_2 - layer_all.log_partition(eta), dim=0)

        return SymbCfgFactory(
            layer_cls=ParameterizedConstantLayer, reparam=UnaryReparam(symb_cfg.reparam, func=_func)
        )

    @classmethod
    def get_partial(
        cls, symb_cfg: SymbLayerCfg[Self], *, order: int = 1, var_idx: int = 0, ch_idx: int = 0
    ) -> SymbCfgFactory[InputLayer]:
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
            SymbCfgFactory[InputLayer]: The symbolic config for the partial differential w.r.t. \
                the given channel of the given variable.
        """
        # TODO: support nested ProdEFLayer (3 or more products)
        # TODO: how to check continuous/discrete distribution
        # CAST: kwargs.get gives Any.
        # IGNORE: Unavoidable for kwargs.
        if CategoricalLayer in (
            cast(
                SymbLayerCfg[ExpFamilyLayer],
                symb_cfg.layer_kwargs.get("ef1_cfg"),  # type: ignore[misc]
            ).layer_cls,
            cast(
                SymbLayerCfg[ExpFamilyLayer],
                symb_cfg.layer_kwargs.get("ef2_cfg"),  # type: ignore[misc]
            ).layer_cls,
        ):
            raise TypeError("Cannot differentiate over discrete variables.")

        return super().get_partial(symb_cfg, order=order, var_idx=var_idx, ch_idx=ch_idx)

    # NOTE: get_product is inherited from ExpFamilyLayer. For cascaded multiplication this will lead
    #       to a binary tree of ProdEFLayer.
