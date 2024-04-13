from typing import Callable, Dict, Iterable, Optional, Type, Union

from cirkit.symbolic.symb_circuit import SymbCircuit, SymbCircuitOperation, SymbCircuitOperator
from cirkit.symbolic.symb_layers import (
    SymbConstantLayer,
    SymbExpFamilyLayer,
    SymbInputLayer,
    SymbLayer,
    SymbLayerOperation,
    SymbLayerOperator,
    SymbProdLayer,
    SymbSumLayer,
)
from cirkit.utils import Scope


def integrate_ef_layer(
    sl: SymbExpFamilyLayer, scope: Optional[Iterable[int]] = None
) -> SymbConstantLayer:
    # Symbolically integrate an exponential family layer
    return SymbConstantLayer(sl.scope, sl.num_units, sl.num_channels, value=1.0)


def integrate_input_layer(
    sl: SymbInputLayer, scope: Optional[Iterable[int]] = None
) -> SymbConstantLayer:
    # Fallback functional as to implement symbolic integration over any other symbolic input layer
    # Note that the operator data structure will store the relevant information,
    # i.e., the layer itself as an operand and the variables to integrate as metadata
    scope = Scope(scope) if scope is not None else sl.scope
    return SymbConstantLayer(
        sl.scope,
        sl.num_units,
        sl.num_channels,
        operation=SymbLayerOperation(
            SymbLayerOperator.INTEGRATION, operands=(sl,), metadata=dict(scope=scope)
        ),
    )


def integrate(
    symb_circuit: SymbCircuit,
    /,
    *,
    scope: Optional[Iterable[int]] = None,
    registry: Optional[
        Dict[Type[SymbInputLayer], Callable[[SymbInputLayer], SymbConstantLayer]]
    ] = None,
) -> SymbCircuit:
    assert (
        symb_circuit.is_smooth and symb_circuit.is_decomposable
    ), "Only smooth and decomposable circuits can be efficiently integrated."
    scope = Scope(scope) if scope is not None else symb_circuit.scope
    assert (
        scope | symb_circuit.scope
    ) == symb_circuit.scope, (
        "The variables scope to integrate must be a subset of the scope of the circuit"
    )

    # The default registry the compiler will use to integrate input layers
    integrate_registry = {
        SymbExpFamilyLayer: integrate_ef_layer,
        SymbInputLayer: integrate_input_layer,
    }

    # Update with the registry specified by the user (if any)
    if registry is not None:
        integrate_registry.update(registry)

    symbc_to_integral: Dict[SymbLayer, SymbLayer] = {}
    for sl in symb_circuit.layers:
        if isinstance(sl, SymbInputLayer) and sl.scope & scope:  # input layers -> integrate
            if not (sl.scope <= scope):
                raise NotImplementedError(
                    "Multivariate integration of proper subsets of variables is not implemented"
                )
            # Retrieve the transformation function from the registry
            # If none is found, then use the fallback one that works for all input layers
            transform_func = integrate_registry.get(type(sl), integrate_registry[SymbInputLayer])
            symbc_to_integral[sl] = transform_func(sl)
        else:  # sum or product layer -> just make a copy and set the operation attribute
            assert isinstance(sl, (SymbSumLayer, SymbProdLayer))
            integral_sl_inputs = [symbc_to_integral[isl] for isl in sl.inputs]
            integral_sl: Union[SymbSumLayer, SymbProdLayer] = type(sl)(
                sl.scope,
                sl.num_units,
                operation=SymbLayerOperation(
                    operator=SymbLayerOperator.INTEGRATION, operands=(sl,)
                ),
                inputs=integral_sl_inputs,
            )
            symbc_to_integral[sl] = integral_sl

    # Construct the integral symbolic circuit and set the integration operation metadata
    return type(symb_circuit)(
        symb_circuit.region_graph,
        symbc_to_integral.values(),
        operation=SymbCircuitOperation(
            operator=SymbCircuitOperator.INTEGRATION,
            operands=(symb_circuit,),
            metadata=dict(scope=scope),
        ),
    )


def multiply(
    lhs_symb_circuit: SymbCircuit,
    rhs_symb_circuit: SymbCircuit,
) -> SymbCircuit:
    pass


def differentiate(symb_circuit: SymbCircuit) -> SymbCircuit:
    pass
