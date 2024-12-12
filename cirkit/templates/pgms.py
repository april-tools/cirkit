from collections.abc import Sequence

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, Layer, SumLayer
from cirkit.symbolic.parameters import ParameterFactory
from cirkit.templates.utils import InputLayerFactory
from cirkit.utils.scope import Scope


def from_hmm(
    ordering: Sequence[int],
    input_factory: InputLayerFactory,
    weight_factory: ParameterFactory | None = None,
    num_channels: int = 1,
    num_units: int = 1,
    num_classes: int = 1,
) -> Circuit:
    """Construct a symbolic circuit mimicking a hidden markov model (HMM) of
      a given variable ordering. Product Layers are of type
      [HadamardLayer][cirkit.symbolic.layers.HadamardLayer], and sum layers are of type
      [SumLayer][cirkit.symbolic.layers.SumLayer].

    Args:
        ordering: The input order of variables of the HMM.
        input_factory: A factory that builds input layers.
        weight_factory: The factory to construct the weight of sum layers. It can be None,
            or a parameter factory, i.e., a map from a shape to a symbolic parameter.
        num_channels: The number of channels for each variable.
        num_units: The number of sum units per sum layer.
        num_classes: The number of output classes.

    Returns:
        Circuit: A symbolic circuit.

    Raises:
        ValueError: order must consists of consistent numbers, starting from 0.
    """
    if max(ordering) != len(ordering) - 1 or min(ordering):
        raise ValueError("The 'ordering' of variables is not valid")

    layers: list[Layer] = []
    in_layers: dict[Layer, list[Layer]] = {}

    input_sl = input_factory(Scope([ordering[0]]), num_units, num_channels)
    layers.append(input_sl)
    sum_sl = SumLayer(num_units, num_units, weight_factory=weight_factory)
    layers.append(sum_sl)
    in_layers[sum_sl] = [input_sl]

    # Loop over the number of variables
    for i in range(1, len(ordering)):
        last_dense = layers[-1]

        input_sl = input_factory(Scope([ordering[i]]), num_units, num_channels)
        layers.append(input_sl)
        prod_sl = HadamardLayer(num_units, 2)
        layers.append(prod_sl)
        in_layers[prod_sl] = [last_dense, input_sl]

        num_units_out = num_units if i != len(ordering) - 1 else num_classes
        sum_sl = SumLayer(
            num_units,
            num_units_out,
            weight_factory=weight_factory,
        )
        layers.append(sum_sl)
        in_layers[sum_sl] = [prod_sl]

    return Circuit(num_channels, layers, in_layers, [layers[-1]])
