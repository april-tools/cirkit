from typing import Optional
from typing_extensions import Self  # FUTURE: in typing from 3.11

import torch
from torch import Tensor

from cirkit.new.layers.input.input import InputLayer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbLayerCfg


class ConstantLayer(InputLayer):
    """The constant input layer, with no parameters."""

    # We still accept any Reparameterization instance for reparam, but it will be ignored.
    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        reparam: Optional[Reparameterization] = None,
        const_value: float,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (Optional[Reparameterization], optional): Ignored. This layer has no params. \
                Defaults to None.
            const_value (float): The constant value, in linear space.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=None,
        )

        self.const_value = const_value

    def reset_parameters(self) -> None:
        """Do nothing, as constant layers do not have parameters."""

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, K).

        Returns:
            Tensor: The output of this layer, shape (*B, K).
        """
        return (
            self.comp_space.from_linear(torch.tensor(self.const_value))
            .to(x)
            .expand(*x.shape[1:-1], self.num_output_units)
        )

    @classmethod
    def get_integral(cls, symb_cfg: SymbLayerCfg[Self]) -> SymbLayerCfg[InputLayer]:
        """Get the symbolic config to construct the definite integral of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.

        Raises:
            ValueError: When const_value != 0, in which case the integral is infinity.

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the integral.
        """
        # IGNORE: Unavoidable for kwargs.
        assert isinstance(
            const_value := symb_cfg.layer_kwargs.get("const_value", None),  # type: ignore[misc]
            (float, int),
        ), "Mismatched kwargs for this layer."

        if const_value:
            raise ValueError(
                "The definite integral of ConstantLayer with const_value!=0 is infinity."
            )

        # IGNORE: Unavoidable for kwargs.
        return SymbLayerCfg(
            layer_cls=ConstantLayer, layer_kwargs={"const_value": 0.0}  # type: ignore[misc]
        )

    @classmethod
    def get_partial(
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

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the partial differential w.r.t. the \
                given channel of the given variable.
        """
        assert order >= 0, "The order of differential must be non-negative."
        if not order:
            return symb_cfg

        # IGNORE: Unavoidable for kwargs.
        return SymbLayerCfg(
            layer_cls=ConstantLayer, layer_kwargs={"const_value": 0.0}  # type: ignore[misc]
        )

    @classmethod
    def get_product(
        cls, self_symb_cfg: SymbLayerCfg[Self], other_symb_cfg: SymbLayerCfg[InputLayer]
    ) -> SymbLayerCfg[InputLayer]:
        """Get the symbolic config to construct the product of this input layer with the other \
        input layer.

        TODO: scaling of other layers.

        Args:
            self_symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.
            other_symb_cfg (SymbLayerCfg[InputLayer]): The symbolic config for the other layer, \
                must be of InputLayer.

        Raises:
            NotImplementedError: When "not-yet-implemented feature" is invoked.

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the product of the two input layers.
        """
        if issubclass(other_symb_cfg.layer_cls, ConstantLayer):
            # IGNORE: Unavoidable for kwargs.
            return SymbLayerCfg(
                layer_cls=ConstantLayer,
                layer_kwargs={
                    "const_value": self_symb_cfg.layer_kwargs["const_value"]  # type: ignore[misc]
                    * other_symb_cfg.layer_kwargs["const_value"]  # type: ignore[misc]
                },
            )

        raise NotImplementedError(
            "Product for constant input layer and other input layers not implemented."
        )
