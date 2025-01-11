import numpy as np

from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.symbolic.layers import CategoricalLayer, InputLayer
from cirkit.symbolic.parameters import Parameter, TensorParameter
from cirkit.templates.utils import InputLayerFactory
from cirkit.utils.scope import Scope


def default_literal_input_factory(negated: bool = False) -> InputLayerFactory:
    """Input factory for a boolean logic circuit input realized using a
    Categorical Layer constantly parametrized by a tensor [x, y] where x is
    the probability of being False and y the probability of being True.

    Args:
        negated (bool): If True returns the input factory for the negated literal,
            else return a regular input factory.

    Returns:
        InputLayerFactory: The input layer factory.
    """

    def input_factory(scope: Scope, num_units: int) -> InputLayer:
        param = np.array([1.0, 0.0]) if negated else np.array([0.0, 1.0])
        initializer = ConstantTensorInitializer(param)
        return CategoricalLayer(
            scope,
            num_categories=2,
            num_output_units=num_units,
            probs=Parameter.from_input(TensorParameter(1, 1, 2, initializer=initializer)),
        )

    return input_factory
