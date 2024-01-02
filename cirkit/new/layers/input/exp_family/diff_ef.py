from typing import Any, Dict, Type, cast
from typing_extensions import Self  # FUTURE: in typing from 3.11

from torch import Tensor

from cirkit.new.layers.input.exp_family.exp_family import ExpFamilyLayer
from cirkit.new.layers.input.input import InputLayer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils import batch_high_order_at
from cirkit.new.utils.type_aliases import SymbLayerCfg


class DiffEFLayer(InputLayer):
    """The partial differential for Exponential Family distribution layers.

    Exponential Family dist:
        f(x) = exp(eta · T(x) - log_h(x) + A(eta)) = exp(g(x)),
    where g(x) is log-prob.

    Differentials:
        f'(x) = f(x)g'(x);
        f''(x) = f(x)(g'(x)^2 + g''(x)).
    """

    # DISABLE: It's designed to have these arguments.
    # IGNORE: Unavoidable for kwargs.
    def __init__(  # type: ignore[misc]  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        reparam: Reparameterization,
        ef_cls: Type[ExpFamilyLayer],
        ef_kwargs: Dict[str, Any],
        order: int,
        var_idx: int,
        ch_idx: int,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (Reparameterization): The reparameterization for layer parameters.
            ef_cls (Type[ExpFamilyLayer]): The class of ExpFamilyLayer to differentiate.
            ef_kwargs (Dict[str,Any]): The kwargs of ExpFamilyLayer to differentiate.
            order (int): The order of differentiation.
            var_idx (int): The variable to diffrentiate. The idx is counted within this layer's \
                scope but not global variable id.
            ch_idx (int): The channel of variable to diffrentiate.
        """
        assert (  # TODO: however order==0 actually also works here.
            order > 0
        ), "The order must be positive for DiffEFLayer (use ExpFamilyLayer for order=0)."
        assert (
            0 <= var_idx < arity and 0 <= ch_idx < num_input_units
        ), "The variable/channel to differentiate is out of bound."
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        # IGNORE: Unavoidable for kwargs.
        self.ef = ef_cls(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
            **ef_kwargs,  # type: ignore[misc]
        )
        # ExpFamilyLayer already invoked reset_parameters().

        self.order = order
        self.var_idx = var_idx
        self.ch_idx = ch_idx

    def reset_parameters(self) -> None:
        """Do nothing, as the parameters belong to the wrapped EF layer."""

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, K).

        Raises:
            NotImplementedError: When "not-yet-implemented feature" is invoked -- order > 2.

        Returns:
            Tensor: The output of this layer, shape (*B, K).
        """
        eta_value = self.ef.params()
        g_diffs = batch_high_order_at(
            lambda x: self.ef.log_prob(eta_value, x),
            x,
            [self.var_idx, ..., self.ch_idx],
            order=self.order,
        )
        log_p = g_diffs[0]  # g(x) = log_p.

        # order >= 1 is asserted in __init__.
        if self.order == 1:
            # g_factor: G(x) = f'(x)/f(x) = g'(x).
            g_factor = g_diffs[1]
        elif self.order == 2:
            # g_factor: G(x) = f''(x)/f(x) = g'(x)^2 + g''(x).
            g_factor = g_diffs[1] ** 2 + g_diffs[2]  # type: ignore[misc]  # TODO: __pow__ issue
        else:
            # TODO: or no specific for EF, but generalize to all input layers using jac_functional?
            raise NotImplementedError("order>2 is not yet implemented for DiffEFLayer.")

        return self.comp_space.mul(
            self.comp_space.from_log(log_p), self.comp_space.from_linear(g_factor)
        )

    # IGNORE: SymbLayerCfg contains Any.
    @classmethod
    def get_integral(  # type: ignore[misc]
        cls, symb_cfg: SymbLayerCfg[Self]
    ) -> SymbLayerCfg[InputLayer]:
        """Get the symbolic config to construct the definite integral of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer. Unused here.

        Raises:
            NotImplementedError: When "not-yet-implemented feature" is invoked.

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the integral.
        """
        raise NotImplementedError("The integral of DiffEFLayer is not yet defined.")

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
            NotImplementedError: When "not-yet-implemented feature" is invoked.

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the partial differential w.r.t. the \
                given channel of the given variable.
        """
        # TODO: duplicate code?
        assert order >= 0, "The order of differential must be non-negative."
        if not order:
            # TODO: variance issue
            return cast(SymbLayerCfg[InputLayer], symb_cfg)  # type: ignore[misc]

        # TODO: for same var_idx and ch_idx, can reuse the same symb_cfg with only order increased.

        raise NotImplementedError("The partial differential of DiffEFLayer is not yet defined.")
