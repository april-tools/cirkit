from typing import TYPE_CHECKING, Dict, Iterable, Optional

from cirkit.new.symbolic.symbolic_layer import SymbolicInputLayer, SymbolicLayer
from cirkit.new.utils import OrderedSet, Scope

if TYPE_CHECKING:  # Only imported for static type checking but not runtime, to avoid cyclic import.
    from cirkit.new.symbolic.symbolic_circuit import SymbolicTensorizedCircuit


def integrate(
    self: "SymbolicTensorizedCircuit", *, scope: Optional[Iterable[int]] = None
) -> "SymbolicTensorizedCircuit":
    """Integrate the circuit over the variables specified by the given scope.

    Args:
        self (SymbolicTensorizedCircuit): The circuit to integrate.
        scope (Optional[Iterable[int]], optional): The scope over which to integrate, or None for \
            the whole scope of the circuit. Defaults to None.

    Returns:
        SymbolicTensorizedCircuit: The circuit giving the integral.
    """
    scope = Scope(scope) if scope is not None else self.scope

    integral = object.__new__(self.__class__)
    # Skip integral.__init__ and customize initialization as below.

    integral.region_graph = self.region_graph
    integral.scope = self.scope  # TODO: is this a good definition?
    integral.num_vars = self.num_vars
    integral.is_smooth = self.is_smooth
    integral.is_decomposable = self.is_decomposable
    integral.is_structured_decomposable = self.is_structured_decomposable
    integral.is_omni_compatible = self.is_omni_compatible
    integral.num_classes = self.num_classes

    # Disable: It's intended to use _layers from SymbolicTensorizedCircuit.
    # pylint: disable=protected-access

    integral._layers = OrderedSet()

    self_to_integral: Dict[SymbolicLayer, SymbolicLayer] = {}  # Map between two SymbC.

    for self_layer in self._layers:
        integral_layer: SymbolicLayer
        # Ignore: SymbolicInputLayer contains Any.
        # Ignore: Unavoidable for kwargs.
        if (
            isinstance(self_layer, SymbolicInputLayer)  # type: ignore[misc]
            and self_layer.scope & scope
        ):
            assert (
                self_layer.scope <= scope
            ), "The scope of an input layer must be either all marginalized or all not."
            layer_cls, layer_kwargs = self_layer.layer_cls.get_integral(  # type: ignore[misc]
                **self_layer.layer_kwargs  # type: ignore[misc]
            )
            integral_layer = SymbolicInputLayer(
                self_layer.rg_node,
                (),
                num_units=self_layer.num_units,
                layer_cls=layer_cls,
                layer_kwargs=layer_kwargs,  # type: ignore[misc]
                reparam=self_layer.reparam,  # Reuse the same reparam if the integral needs it.
            )
        else:
            integral_layer = self_layer.__class__(
                self_layer.rg_node,
                (self_to_integral[self_layer_in] for self_layer_in in self_layer.inputs),
                num_units=self_layer.num_units,
                layer_cls=self_layer.layer_cls,
                layer_kwargs=self_layer.layer_kwargs,  # type: ignore[misc]
                reparam=self_layer.reparam,  # Reuse the same reparam to share params.
            )
        integral._layers.append(integral_layer)
        self_to_integral[self_layer] = integral_layer

    # pylint: enable=protected-access

    return integral
