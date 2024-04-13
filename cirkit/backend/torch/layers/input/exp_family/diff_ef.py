from typing import Optional

from torch import Tensor

from cirkit.backend.torch.layers.input import ExpFamilyLayer
from cirkit.backend.torch.reparams import Reparameterization
from cirkit.layers.input import InputLayer
from cirkit.utils import batch_high_order_at


class DiffEFLayer(InputLayer):
    """The partial differential for Exponential Family distribution layers.

    Exponential Family dist:
        p(x) = exp(η · T(x) + log_h(x) - A(η)) = exp(log_p(x)).

    Differentials:
        p'(x) = p(x)log_p'(x);
        p''(x) = p(x)(log_p'(x)^2 + log_p''(x)).
    """

    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        reparam: Optional[Reparameterization] = None,
        ef_cfg: SymbLayerCfg[ExpFamilyLayer],
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
            reparam (Optional[Reparameterization], optional): Ignored. This layer has no params \
                itself but only holds the params through the ef_cfg passed in. Defaults to None.
            ef_cfg (SymbLayerCfg[ExpFamilyLayer]): The config of ExpFamilyLayer to differentiate, \
                should include a reference to a concretized SymbL for EF.
            order (int): The order of differentiation.
            var_idx (int): The variable to diffrentiate. The idx is counted within this layer's \
                scope but not global variable id.
            ch_idx (int): The channel of variable to diffrentiate.
        """
        assert (
            order > 0
        ), "The order must be positive for DiffEFLayer (use ExpFamilyLayer for order=0)."
        assert (
            0 <= var_idx < arity and 0 <= ch_idx < num_input_units
        ), "The variable/channel to differentiate is out of bound."
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=None,
        )

        assert (
            ef := ef_cfg.symb_layer.concrete_layer
        ) is not None, (
            "There should be a concrete Layer corresponding to the SymbLayerCfg at this stage."
        )
        self.ef = ef

        self.order = order
        self.var_idx = var_idx
        self.ch_idx = ch_idx

    @property
    def _default_initializer_(self) -> None:
        """The default inplace initializer for the parameters of this layer.

        No initialization, as DiffEFLayer's parameters should come from other layers.
        """
        return None

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Raises:
            NotImplementedError: When "not-yet-implemented feature" is invoked -- order > 2.

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        logp_diffs = batch_high_order_at(
            self.ef.log_prob, x, [self.var_idx, ..., self.ch_idx], order=self.order
        )
        log_p = logp_diffs[0]

        # order >= 1 is asserted in __init__.
        if self.order == 1:
            # factor = p'(x)/p(x) = log_p'(x).
            factor = logp_diffs[1]
        elif self.order == 2:
            # factor = p''(x)/p(x) = log_p'(x)^2 + log_p''(x).
            factor = logp_diffs[1] ** 2 + logp_diffs[2]  # type: ignore[misc]  # TODO: __pow__ issue
        else:
            # TODO: or no specific for EF, but generalize to all input layers using jac_functional?
            raise NotImplementedError("order>2 is not yet implemented for DiffEFLayer.")

        return self.comp_space.mul(
            self.comp_space.from_log(log_p), self.comp_space.from_linear(factor)
        )
