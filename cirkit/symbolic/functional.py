from collections import defaultdict
from typing import Dict, Iterable, Optional, Union, List

from cirkit.symbolic.registry import SymbOperatorRegistry
from cirkit.symbolic.symb_circuit import SymbCircuit, SymbCircuitOperation, SymbCircuitOperator
from cirkit.symbolic.symb_layers import (
    SymbInputLayer,
    SymbLayer,
    SymbLayerOperation,
    SymbProdLayer,
    SymbSumLayer, SymbLayerOperator,
)
from cirkit.utils import Scope
from cirkit.utils.exceptions import StructuralPropertyError


def integrate(
    sc: SymbCircuit,
    registry: SymbOperatorRegistry,
    scope: Optional[Iterable[int]] = None
) -> SymbCircuit:
    if not sc.is_smooth or not sc.is_decomposable:
        raise StructuralPropertyError(
            "Only smooth and decomposable circuits can be efficiently integrated.")
    scope = Scope(scope) if scope is not None else sc.scope
    if (scope | sc.scope) != sc.scope:
        raise ValueError("The variables scope to integrate must be a subset of the scope of the circuit")

    # Mapping the symbolic circuit layers with the layers of the new circuit to build
    map_layers: Dict[SymbLayer, SymbLayer] = {}

    # For each new layer, keep track of (i) its inputs and (ii) the layers it feeds
    in_layers: Dict[SymbLayer, List[SymbLayer]] = defaultdict(list)
    out_layers: Dict[SymbLayer, List[SymbLayer]] = defaultdict(list)

    for sl in sc.layers:
        # Input layers get integrated over
        if isinstance(sl, SymbInputLayer) and sl.scope & scope:
            if not (sl.scope <= scope):
                raise NotImplementedError(
                    "Multivariate integration of proper subsets of variables is not implemented"
                )
            # Retrieve the integration rule from the registry and apply it
            if registry.has_rule(SymbLayerOperator.INTEGRATION, type(sl)):
                func = registry.retrieve_rule(SymbLayerOperator.INTEGRATION, type(sl))
            else:  # Use a fallback rule that is not a specialized one
                func = registry.retrieve_rule(SymbLayerOperator.INTEGRATION, SymbInputLayer)
            map_layers[sl] = func(sl)
        else:  # Sum/product layers are simply copied
            assert isinstance(sl, (SymbSumLayer, SymbProdLayer))
            new_sl_inputs = [map_layers[isl] for isl in sc.layer_inputs(sl)]
            new_sl: Union[SymbSumLayer, SymbProdLayer] = type(sl)(
                sl.scope,
                sl.num_units,
                arity=sl.arity,
                operation=SymbLayerOperation(
                    operator=SymbLayerOperator.NOP, operands=(sl,)
                )
            )
            map_layers[sl] = new_sl
            in_layers[new_sl] = new_sl_inputs
            for isl in new_sl_inputs:
                out_layers[isl] = new_sl_inputs

    # Construct the integral symbolic circuit and set the integration operation metadata
    return SymbCircuit(
        sc.scope,
        list(map_layers.values()),
        in_layers,
        out_layers,
        operation=SymbCircuitOperation(
            operator=SymbCircuitOperator.INTEGRATION,
            operands=(sc,),
            metadata=dict(scope=scope),
        ),
    )


def multiply(
    lhs_sc: SymbCircuit,
    rhs_sc: SymbCircuit,
    registry: SymbOperatorRegistry
) -> SymbCircuit:
    if not lhs_sc.is_compatible(rhs_sc):
        raise StructuralPropertyError(
            "Only compatible circuits can be multiplied into decomposable circuits.")
    ...


def differentiate(sc: SymbCircuit, registry: SymbOperatorRegistry) -> SymbCircuit:
    if not sc.is_smooth or not sc.is_decomposable:
        raise StructuralPropertyError(
            "Only smooth and decomposable circuits can be efficiently differentiated.")
    ...
