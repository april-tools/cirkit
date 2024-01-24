from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:  # Only imported for static type checking but not runtime, to avoid cyclic import.
    from cirkit.new.model.tensorized_circuit import TensorizedCircuit


def integrate(
    self: "TensorizedCircuit", *, scope: Optional[Iterable[int]] = None
) -> "TensorizedCircuit":
    """Integrate the circuit over the variables specified by the given scope.

    Args:
        self (TensorizedCircuit): The circuit to integrate.
        scope (Optional[Iterable[int]], optional): The scope over which to integrate, or None for \
            the whole scope of the circuit. Defaults to None.

    Returns:
        TensorizedCircuit: The circuit giving the definite integral.
    """
    return self.__class__(self.symb_circuit.integrate(scope=scope))


def differentiate(self: "TensorizedCircuit", *, order: int = 1) -> "TensorizedCircuit":
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
    return self.__class__(self.symb_circuit.differentiate(order=order))


def product(self: "TensorizedCircuit", other: "TensorizedCircuit") -> "TensorizedCircuit":
    """Perform product between two circuits over their intersected scope.

    Args:
        self (TensorizedCircuit): The first circuit to perform product.
        other (TensorizedCircuit): The second circuit to perform product.

    Returns:
        SymbolicTensorizedCircuit: The circuit product.
    """
    return self.__class__(self.symb_circuit.product(other.symb_circuit))
