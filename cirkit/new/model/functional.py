from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:  # Only imported for static type checking but not runtime, to avoid cyclic import.
    from cirkit.new.model.tensorized_circuit import TensorizedCircuit


def integrate(
    self: "TensorizedCircuit", /, *, scope: Optional[Iterable[int]] = None
) -> "TensorizedCircuit":
    """Integrate the circuit over the variables specified by the given scope.

    Args:
        self (TensorizedCircuit): The circuit to integrate.
        scope (Optional[Iterable[int]], optional): The scope over which to integrate, or None for \
            the whole scope of the circuit. Defaults to None.

    Returns:
        TensorizedCircuit: The circuit giving the definite integral.
    """
    return type(self)(self.symb_circuit.integrate(scope=scope))


def differentiate(self: "TensorizedCircuit", /, *, order: int = 1) -> "TensorizedCircuit":
    """Differentiate the circuit w.r.t. each variable (i.e. total differentiate) to the given order.

    NOTE: Each output layer will be expanded to (layer.num_vars * num_channels + 1) output layers \
          consecutive in the layer container, with all-but-last layers reshapable to \
          (layer.num_vars, num_channels) calculating the partial differential w.r.t. each variable \
          in the layer's scope and each channel in the variables, and the last one calculating the \
          original, which is copied (but not referenced) from the original circuit, i.e., the same \
          SymbolicLayer object will not be reused in a different SymbolicTensorizedCircuit.

    Args:
        self (TensorizedCircuit): The circuit to differentiate.
        order (int, optional): The order of differentiation. Defaults to 1.

    Returns:
        TensorizedCircuit: The circuit giving the (total) differential.
    """
    return type(self)(self.symb_circuit.differentiate(order=order))


def multiply(self: "TensorizedCircuit", other: "TensorizedCircuit", /) -> "TensorizedCircuit":
    """Multiply two circuits over the intersection of their scopes.

    Args:
        self (TensorizedCircuit): The left operand circuit (the self circuit).
        other (TensorizedCircuit): The right operand circuit (the other circuit).

    Returns:
        SymbolicTensorizedCircuit: The product circuit.
    """
    return type(self)(self.symb_circuit.multiply(other.symb_circuit))
