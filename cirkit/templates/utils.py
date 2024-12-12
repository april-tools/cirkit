from typing import Protocol

from cirkit.symbolic.layers import InputLayer, SumLayer, ProductLayer
from cirkit.utils.scope import Scope


class InputLayerFactory(Protocol):  # pylint: disable=too-few-public-methods
    """The protocol of a factory that constructs input layers."""

    def __call__(self, scope: Scope, num_units: int, num_channels: int) -> InputLayer:
        """Constructs an input layer.

        Args:
            scope: The scope of the layer.
            num_units: The number of input units composing the layer.
            num_channels: The number of channel variables.

        Returns:
            InputLayer: An input layer.
        """


class SumLayerFactory(Protocol):  # pylint: disable=too-few-public-methods
    """The protocol of a factory that constructs sum layers."""

    def __call__(self, num_input_units: int, num_output_units: int) -> SumLayer:
        """Constructs a sum layer.

        Args:
            num_input_units: The number of units in each layer that is an input.
            num_output_units: The number of sum units in the layer.

        Returns:
            SumLayer: A sum layer.
        """


class ProductLayerFactory(Protocol):  # pylint: disable=too-few-public-methods
    """The protocol of a factory that constructs product layers."""

    def __call__(self, num_input_units: int, arity: int) -> ProductLayer:
        """Constructs a product layer.

        Args:
            num_input_units: The number of units in each layer that is an input.
            arity: The number of input layers.

        Returns:
            ProductLayer: A product layer.
        """
