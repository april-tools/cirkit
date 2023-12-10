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
        TensorizedCircuit: The circuit giving the integral.
    """
    return self.__class__(self.symb_circuit.integrate(scope=scope), num_channels=self.num_channels)
