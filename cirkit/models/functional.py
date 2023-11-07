from cirkit.layers.input.constant import ConstantLayer
from cirkit.models import TensorizedPC


def integrate(circuit: TensorizedPC) -> TensorizedPC:
    """Integrate a circuit over all the variables it is defined on.

    This returns another circuit sharing the parameters with the given circuit. The input of its
    forward method can be None, as the function it computes does not depend on the input.

    Args:
        circuit: A tensorized circuit.

    Returns:
        TensorizedPC: A tensorized circuit that computes its integral.
    """
    constant_layer = ConstantLayer(value=circuit.input_layer.integrate)
    new_circuit = TensorizedPC(
        input_layer=constant_layer,
        scope_layer=circuit.scope_layer,
        bookkeeping=circuit.bookkeeping,
        inner_layers=circuit.inner_layers,
        integral_input_layer=circuit.integral_input_layer,
    )
    return new_circuit
