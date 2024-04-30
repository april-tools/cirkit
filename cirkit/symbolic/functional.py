import itertools
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from cirkit.symbolic.circuit import (
    Circuit,
    CircuitBlock,
    CircuitOperation,
    CircuitOperator,
    StructuralPropertyError,
)
from cirkit.symbolic.layers import (
    InputLayer,
    Layer,
    LayerOperation,
    PlaceholderParameter,
    ProductLayer,
    SumLayer,
)
from cirkit.symbolic.registry import OPERATOR_REGISTRY, OperatorRegistry
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

    # Use the registry in the current context, if not specified otherwise
    if registry is None:
        registry = OPERATOR_REGISTRY.get()

    # Mapping the symbolic circuit layers with blocks of circuit layers
    map_layers: Dict[Layer, CircuitBlock] = {}

    # For each new circuit block, keep track of (i) its inputs and (ii) the blocks it feeds
    in_blocks: Dict[CircuitBlock, List[CircuitBlock]] = defaultdict(list)
    out_blocks: Dict[CircuitBlock, List[CircuitBlock]] = defaultdict(list)

    ordering = sc.layers_topological_ordering()
    for sl in ordering:
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
            int_block = func(sl)
            map_layers[sl] = int_block
            continue
        assert isinstance(
            sl, (SumLayer, ProductLayer)
        ), "Symbolic inner layers must be either sum or product layers"
        # Sum/product layers are simply copied
        # Placeholders are used to keep track of referenced parameters
        learnable_parameters = {
            pname: PlaceholderParameter(sl, pname) for pname in sl.parameters.keys()
        }
        int_block = CircuitBlock.from_layer(type(sl)(**sl.config, **learnable_parameters))
        map_layers[sl] = int_block
        int_block_ins = [map_layers[isl] for isl in sc.layer_inputs(sl)]
        in_blocks[int_block] = int_block_ins
        for bi in int_block_ins:
            out_blocks[bi].append(int_block)

    # Construct the integral symbolic circuit and set the integration operation metadata
    return Circuit.from_operation(
        sc.scope,
        sc.num_channels,
        list(map_layers.values()),
        in_blocks,
        out_blocks,
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

    # Use the registry in the current context, if not specified otherwise
    if registry is None:
        registry = OPERATOR_REGISTRY.get()

    # Map from pairs of layers to their product circuit block
    map_layers: Dict[Tuple[Layer, Layer], CircuitBlock] = {}

    # For each new circuit block, keep track of (i) its inputs and (ii) the blocks it feeds
    in_blocks: Dict[CircuitBlock, List[CircuitBlock]] = defaultdict(list)
    out_blocks: Dict[CircuitBlock, List[CircuitBlock]] = defaultdict(list)

    # Get the first layers to multiply, from the outputs
    to_multiply = []
    for lhs_out, rhs_out in itertools.product(lhs_sc.output_layers, rhs_sc.output_layers):
        to_multiply.append((lhs_out, rhs_out))

    # Using a stack in place of recursion for better memory efficiency and debugging
    while to_multiply:
        pair = to_multiply[-1]
        if pair in map_layers:
            to_multiply.pop()
            continue
        lhs_layer, rhs_layer = pair
        if type(lhs_layer) != type(rhs_layer):  # pylint: disable=unidiomatic-typecheck
            raise NotImplementedError(
                "The multiplication of circuits with different layers or region graphs has not been implemented yet"
            )

        lhs_inputs = lhs_sc.layer_inputs(lhs_layer)
        rhs_inputs = rhs_sc.layer_inputs(rhs_layer)
        if isinstance(lhs_layer, InputLayer):
            # TODO: generalize product between input and inner layers
            next_to_multiply = []
        elif isinstance(lhs_layer, SumLayer):
            # TODO: generalize product between input and inner layers
            next_to_multiply = list(itertools.product(lhs_inputs, rhs_inputs))
        elif isinstance(lhs_layer, ProductLayer):
            # TODO: generalize product such that it can multiply layers of different arity
            #       this is related to the much more relaxed definition of compatibility between circuits
            assert len(lhs_inputs) == len(rhs_inputs)
            # Sort layers based on the scope, such that we can multiply layers with matching scopes
            lhs_inputs = sorted(lhs_inputs, key=lambda sl: sl.scope)
            rhs_inputs = sorted(rhs_inputs, key=lambda sl: sl.scope)
            next_to_multiply = list(zip(lhs_inputs, rhs_inputs))
        else:
            assert False

        # Check if at least one pair of layers needs to be multiplied before going up in the recursion
        not_yet_multiplied = list(filter(lambda p: p not in map_layers, next_to_multiply))
        if len(not_yet_multiplied) > 0:
            to_multiply.extend(not_yet_multiplied)
            continue

        # In case all the input have been multiplied, then construct the product layer
        prod_signature = type(lhs_layer), type(rhs_layer)
        func = registry.retrieve_rule(LayerOperation.MULTIPLICATION, *prod_signature)
        prod_block = func(lhs_layer, rhs_layer)
        # Make the connections
        prod_block_ins = [map_layers[p] for p in next_to_multiply]
        in_blocks[prod_block] = prod_block_ins
        for bi in prod_block_ins:
            out_blocks[bi].append(prod_block)
        map_layers[pair] = prod_block
        to_multiply.pop()  # Go up in the recursion

    # Construct the product symbolic circuit
    return Circuit.from_operation(
        lhs_sc.scope | rhs_sc.scope,
        lhs_sc.num_channels,
        list(map_layers.values()),
        in_blocks,
        out_blocks,
        operation=CircuitOperation(
            operator=CircuitOperator.MULTIPLICATION, operands=(lhs_sc, rhs_sc)
        ),
    )


def differentiate(sc: Circuit, registry: Optional[OperatorRegistry] = None) -> Circuit:
    if not sc.is_smooth or not sc.is_decomposable:
        raise StructuralPropertyError(
            "Only smooth and decomposable circuits can be efficiently differentiated."
        )

    # Use the registry in the current context, if not specified otherwise
    if registry is None:
        registry = OPERATOR_REGISTRY.get()
