from typing import Literal, Optional
from typing_extensions import Self  # TODO: in typing from 3.11

import torch
from torch import Tensor

from cirkit.new.layers.input.input import InputLayer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbLayerCfg


class ConstantLayer(InputLayer):
    """The constant input layer, with no parameters."""

    # Disable: This __init__ is designed to have these arguments.
    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: Literal[1] = 1,
        reparam: Optional[Reparameterization] = None,
        const_value: float,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (Literal[1], optional): The arity of the layer, must be 1. Defaults to 1.
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

    # Disable/Ignore: It's define with this signature.  # TODO: consider TypedDict?
    @classmethod
    def get_integral(  # type: ignore[misc]  # Ignore: SymbLayerCfg contains Any.
        cls, symb_cfg: SymbLayerCfg[Self]
    ) -> SymbLayerCfg["InputLayer"]:
        """Get the symbolic config to construct the integral of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.

        Raises:
            ValueError: When const_value != 0, in which case the integral is infinity.

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the integral.
        """
        # Ignore: Each step contains Any due to kwargs.
        assert isinstance(
            const_value := symb_cfg.get("layer_kwargs", {}).get(  # type: ignore[misc]
                "const_value", None
            ),
            (float, int),
        ), "Mismatched kwargs for this layer."

        if const_value:
            raise ValueError("The integral of ConstantLayer with const_value != 0 is infinity.")

        return {  # type: ignore[misc]  # Ignore: SymbLayerCfg contains Any.
            "layer_cls": ConstantLayer,
            "layer_kwargs": {"const_value": 0.0},
            "reparam": None,
        }
