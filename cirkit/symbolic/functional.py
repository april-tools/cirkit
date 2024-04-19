from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union

from cirkit.symbolic.circuit import Circuit, CircuitOperation, CircuitOperator
from cirkit.symbolic.layers import (
    InputLayer,
    Layer,
    LayerOperation,
    PlaceholderParameter,
    ProductLayer,
    SumLayer,
)
from cirkit.symbolic.registry import OperatorRegistry
from cirkit.utils.exceptions import StructuralPropertyError
from cirkit.utils.scope import Scope


def integrate(
    sc: Circuit,
    scope: Optional[Iterable[int]] = None,
    registry: Optional[OperatorRegistry] = None,
) -> Circuit:
    # Check for structural properties
    if not sc.is_smooth or not sc.is_decomposable:
        raise StructuralPropertyError(
            "Only smooth and decomposable circuits can be efficiently integrated."
        )

    # Check the variable
    scope = Scope(scope) if scope is not None else sc.scope
    if (scope | sc.scope) != sc.scope:
        raise ValueError(
            "The variables scope to integrate must be a subset of the scope of the circuit"
        )

    # Use the default registry, if not specified otherwise
    if registry is None:
        registry = OperatorRegistry()

    # Mapping the symbolic circuit layers with the layers of the new circuit to build
    map_layers: Dict[Layer, Layer] = {}

    # For each new layer, keep track of (i) its inputs and (ii) the layers it feeds
    in_layers: Dict[Layer, List[Layer]] = defaultdict(list)
    out_layers: Dict[Layer, List[Layer]] = defaultdict(list)

    for sl in sc.layers:
        # Input layers get integrated over
        if isinstance(sl, InputLayer) and sl.scope & scope:
            if not (sl.scope <= scope):
                raise NotImplementedError(
                    "Multivariate integration of proper subsets of variables is not implemented"
                )
            # Retrieve the integration rule from the registry and apply it
            if registry.has_rule(LayerOperation.INTEGRATION, type(sl)):
                func = registry.retrieve_rule(LayerOperation.INTEGRATION, type(sl))
            else:  # Use a fallback rule that is not a specialized one
                func = registry.retrieve_rule(LayerOperation.INTEGRATION, InputLayer)
            map_layers[sl] = func(sl)
            continue
        assert isinstance(
            sl, (SumLayer, ProductLayer)
        ), "Symbolic inner layers must be either sum or product layers"
        # Sum/product layers are simply copied
        # Placeholders are used to keep track of referenced parameters
        new_learnable_parameters = {
            pname: PlaceholderParameter(sl, pname) for pname in sl.learnable_params.keys()
        }
        new_sl: Union[SumLayer, ProductLayer] = type(sl)(**sl.hparams, **new_learnable_parameters)
        map_layers[sl] = new_sl
        new_sl_inputs = [map_layers[isl] for isl in sc.layer_inputs(sl)]
        in_layers[new_sl] = new_sl_inputs
        for isl in new_sl_inputs:
            out_layers[isl] = new_sl

    # Construct the integral symbolic circuit and set the integration operation metadata
    return Circuit(
        sc.scope,
        list(map_layers.values()),
        in_layers,
        out_layers,
        operation=CircuitOperation(
            operator=CircuitOperator.INTEGRATION,
            operands=(sc,),
            metadata=dict(scope=scope),
        ),
    )


def multiply(
    lhs_sc: Circuit, rhs_sc: Circuit, registry: Optional[OperatorRegistry] = None
) -> Circuit:
    if not lhs_sc.is_compatible(rhs_sc):
        raise StructuralPropertyError(
            "Only compatible circuits can be multiplied into decomposable circuits."
        )
    ...


def differentiate(sc: Circuit, registry: Optional[OperatorRegistry] = None) -> Circuit:
    if not sc.is_smooth or not sc.is_decomposable:
        raise StructuralPropertyError(
            "Only smooth and decomposable circuits can be efficiently differentiated."
        )
    ...
