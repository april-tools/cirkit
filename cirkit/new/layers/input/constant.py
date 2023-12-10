from typing import Dict, Literal, Optional, Tuple, Type

import torch
from torch import Tensor

from cirkit.new.layers.input.input import InputLayer
from cirkit.new.reparams import Reparameterization


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
            reparam (Optional[Reparameterization], optional):  Ignored. This layer has no params. \
                Defaults to None.
            const_value (float): The constant value, in linear space.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        self.const_value = const_value

    def reset_parameters(self) -> None:
        """Do nothing as the product layers do not have parameters."""

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
    def get_integral(  # type: ignore[override]  # pylint: disable=arguments-differ
        cls, const_value: float
    ) -> Tuple[Type[InputLayer], Dict[str, float]]:
        """Get the config to construct the integral of the input layer.

        Args:
            const_value (float): The const_value in __init__.

        Raises:
            ValueError: When const_value != 0, in which case the integral is infinity.

        Returns:
            Tuple[Type[InputLayer], Dict[str, float]]: The class of the integral layer and its \
                additional kwargs.
        """
        if const_value:
            raise ValueError("The integral of ConstantLayer with const_value != 0 is infinity.")
        return ConstantLayer, {"const_value": 0.0}
