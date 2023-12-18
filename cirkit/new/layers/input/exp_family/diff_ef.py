from typing import Any, Callable, Dict, Type, cast
from typing_extensions import Self  # TODO: in typing from 3.11

import torch
from torch import Tensor
from torch.autograd.functional import jacobian

from cirkit.new.layers.input.exp_family.exp_family import ExpFamilyLayer
from cirkit.new.layers.input.input import InputLayer
from cirkit.new.reparams import Reparameterization
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

    # Disable: It's designed to have these arguments.
    # Ignore: Unavoidable for kwargs.
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
            reparam=reparam,
        )

        self.ef = ef_cls(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
            **ef_kwargs,  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
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
            NotImplementedError: When "TODO not-yet-implemented feature" is invoked -- order > 2.

        Returns:
            Tensor: The output of this layer, shape (*B, K).
        """
        log_p = self.ef.log_prob(x)

        # order >= 1 is guaranteed for DiffEFLayer.
        g_1st = self.jacobian_functional(self.ef.log_prob)

        if self.order == 1:
            # g_factor: G(x) = f'(x)/f(x) = g'(x).
            g_factor = g_1st(x)
        elif self.order == 2:
            g_2nd = self.jacobian_functional(g_1st)
            # TODO: tensor __pow__ issue
            # g_factor: G(x) = f''(x)/f(x) = g'(x)^2 + g''(x).
            g_factor = g_1st(x) ** 2 + g_2nd(x)  # type: ignore[misc]
        else:
            # TODO: or no specific for EF, but generalize to all input layers using jac_functional?
            raise NotImplementedError("order>2 is not yet implemented for DiffEFLayer.")

        return self.comp_space.mul(
            self.comp_space.from_log(log_p), self.comp_space.from_linear(g_factor)
        )

    def jacobian_functional(self, func: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
        """Apply functional transform on the given func in the shape of forward() to calculate the \
        partial differential (Jacobian) w.r.t. the specified element in input x.

        Args:
            func (Callable[[Tensor], Tensor]): A function that maps shape (H, *B, Ki=C) -> (*B, Ko).

        Returns:
            Callable[[Tensor], Tensor]: The funtion for batched Jacobian of func w.r.t. element \
                (H: var_idx, C: ch_idx) in input x, mapping shape (H, *B, Ki=C) -> (*B, Ko).
        """
        # TODO: pydocstyle requires no blank line after docstring, but black needs one before def.

        # We sum up all *B dims, so that each batch element receive ones as grad;
        # we keep the Ko dim, so that the Jacobian includes it in shape.
        def _summed_func(x: Tensor) -> Tensor:
            """Sum the func output along all batch dimensions.

            Args:
                x (Tensor): The input x to the func, shape (H, *B, Ki=C).

            Returns:
                Tensor: The summed output, shape (Ko,).
            """
            func_x = func(x)
            return func_x.sum(dim=tuple(range(func_x.ndim - 1)))

        # TODO: enable_grad is untyped.
        # Enable grad inside this function, even in no_grad env.
        @torch.enable_grad()  # type: ignore[no-untyped-call]
        def batch_jac_func(x: Tensor) -> Tensor:
            """Calculate the batched Jacobian of the given func.

            Args:
                x (Tensor): The input x, shape (H, *B, Ki=C).

            Returns:
                Tensor: The Jacobian on element (H: var_idx, C: ch_idx) of x, shape (*B, Ko)
            """
            if not x.requires_grad:
                x = x.clone().requires_grad_()  # Enforce grad enabled on x.
            # TODO: is it possible to optimize the following? but for H=1,C=1 there's no difference.

            # TODO: jacobian is untyped.
            jac: Tensor = jacobian(  # type: ignore[no-untyped-call]
                _summed_func, x, create_graph=True
            )  # shape (Ko, H, *B, Ki=C).

            return jac[:, self.var_idx, ..., self.ch_idx].movedim(
                0, -1
            )  # shape (Ko, H: var_idx, *B, Ki=C: ch_idx) -> (Ko, *B) -> (*B, Ko).

        return batch_jac_func

    @classmethod
    def get_integral(  # type: ignore[misc]  # Ignore: SymbLayerCfg contains Any.
        cls, symb_cfg: SymbLayerCfg[Self]
    ) -> SymbLayerCfg[InputLayer]:
        """Get the symbolic config to construct the definite integral of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer. Unused here.

        Raises:
            NotImplementedError: When "TODO not-yet-implemented feature" is invoked.

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the integral.
        """
        raise NotImplementedError("The integral of DiffEFLayer is not yet defined.")

    @classmethod
    def get_partial(  # type: ignore[misc]  # Ignore: SymbLayerCfg contains Any.
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
            NotImplementedError: When "TODO not-yet-implemented feature" is invoked.

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the partial differential w.r.t. the \
                given channel of the given variable.
        """
        # TODO: duplicate code?
        assert order >= 0, "The order of differential must be non-negative."
        if not order:
            # TODO: cast: not sure why SymbLayerCfg[Self] is not SymbLayerCfg[InputLayer] in mypy
            return cast(SymbLayerCfg[InputLayer], symb_cfg)  # type: ignore[misc]

        # TODO: for same var_idx and ch_idx, can reuse the same symb_cfg with only order increased.

        raise NotImplementedError("The partial differential of DiffEFLayer is not yet defined.")
