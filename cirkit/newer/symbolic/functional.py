from typing import Optional, Iterable, Dict, Callable, Type

from cirkit.newer.symbolic.layers import SymbInputLayer
from cirkit.newer.symbolic.layers.input.symb_constant import SymbConstantLayer
from cirkit.newer.symbolic.layers.input.symb_ef import SymbExpFamilyLayer
from cirkit.newer.symbolic.symb_circuit import SymbCircuit
from cirkit.newer.symbolic.symb_op import SymbOperator, SymbLayerOperation
from cirkit.utils import Scope


def integrate_ef_layer(sl: SymbExpFamilyLayer, scope: Optional[Iterable[int]] = None) -> SymbConstantLayer:
    # Symbolically integrate an exponential family layer
    scope = Scope(scope) if scope is not None else sl.scope
    if scope != sl.scope:
        raise NotImplementedError("Integration of proper subsets of variables is not implemented")
    return SymbConstantLayer(
        sl.scope, sl.num_units, sl.num_channels,
        operator=SymbLayerOperation(SymbOperator.INTEGRATION, operands=(), metadata={}),
        value=1.0
    )


def integrate_input_layer(sl: SymbInputLayer, scope: Optional[Iterable[int]] = None) -> SymbConstantLayer:
    # Fallback functional as to implement symbolic integration over any other symbolic input layer
    # Note that the operator data structure will store the relevant information,
    # i.e., the layer itself as an operand and the variables to integrate as metadata
    scope = Scope(scope) if scope is not None else sl.scope
    return SymbConstantLayer(
        sl.scope, sl.num_units, sl.num_channels,
        operator=SymbLayerOperation(
            SymbOperator.INTEGRATION,
            operands=(sl,),
            metadata=dict(scope=scope)
        )
    )


def integrate(
        symb_circuit: SymbCircuit,
        /,
        *,
        scope: Optional[Iterable[int]] = None,
        registry: Optional[Dict[Type[SymbInputLayer], Callable[[SymbInputLayer], SymbConstantLayer]]] = None
) -> SymbCircuit:
    assert (
        symb_circuit.is_smooth and symb_circuit.is_decomposable
    ), "Only smooth and decomposable circuits can be efficiently integrated."
    scope = Scope(scope) if scope is not None else symb_circuit.scope
    assert (scope | symb_circuit.scope) == symb_circuit.scope, \
        "The variables scope to integrate must be a subset of the scope of the circuit"

    # The default registry the compiler will use to integrate input layers
    integrate_registry = {
        SymbExpFamilyLayer: integrate_ef_layer,
        SymbInputLayer: integrate_input_layer
    }

    # Integrate the registry specified by the user (if any) with the default registry
    if registry is None:
        registry = dict()
    registry.update(integrate_registry)

    return symb_circuit
