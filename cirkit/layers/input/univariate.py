from typing import Literal, Optional

from cirkit.layers.input import InputLayer
from cirkit.reparams import Reparameterization


class UnivariateInputLayer(InputLayer):
    """The abstract base class for univariate input layers, with only one variable, one channel."""

    def __init__(
        self,
        *,
        num_input_units: Literal[1] = 1,
        num_output_units: int,
        arity: Literal[1] = 1,
        reparam: Optional[Reparameterization] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (Literal[1], optional): The number of input units, i.e. number of \
                channels for variables, must be 1. Defaults to 1.
            num_output_units (int): The number of output units.
            arity (Literal[1], optional): The arity of the layer, i.e., number of variables in the \
                scope, must be 1. Defaults to 1.
            reparam (Optional[Reparameterization], optional): The reparameterization for layer \
                parameters, can be None if the layer has no params. Defaults to None.
        """
        if num_input_units != 1:
            raise ValueError(
                "UnivariateInputLayer can only have one channel; for more than one, use "
                "Multivariate instead."
            )
        if arity != 1:
            raise ValueError(
                "UnivariateInputLayer can only have one variable; for more than one, use "
                "Multivariate instead."
            )
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )
