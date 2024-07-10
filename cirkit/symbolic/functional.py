import functools
import itertools
import operator
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from cirkit.symbolic.circuit import (
    Circuit,
    CircuitBlock,
    CircuitOperation,
    CircuitOperator,
    StructuralPropertyError,
)
from cirkit.symbolic.layers import InputLayer, Layer, LayerOperation, ProductLayer, SumLayer
from cirkit.symbolic.registry import OPERATOR_REGISTRY, OperatorRegistry
from cirkit.utils.scope import Scope


def merge(scs: Sequence[Circuit], registry: Optional[OperatorRegistry] = None) -> Circuit:
    # Retrieve the number of channels
    assert len(set(sc.num_channels for sc in scs)) == 1
    num_channels = scs[0].num_channels

    # Retrieve the union of the scopes of the circuits
    scope = functools.reduce(operator.ior, map(lambda sc: sc.scope, scs))
    assert scope == scs[0].scope

    # Mapping the symbolic circuit layers with blocks of circuit layers
    layers_to_block: Dict[Layer, CircuitBlock] = {}

    # For each new circuit block, keep track of (i) its inputs and (ii) the blocks it feeds
    blocks: List[CircuitBlock] = []
    in_blocks: Dict[CircuitBlock, List[CircuitBlock]] = {}
    out_blocks: Dict[CircuitBlock, List[CircuitBlock]] = defaultdict(list)

    # Copy the symbolic layers, pick references to parameters and build the blocks
    for sc in scs:
        for sl in sc.topological_ordering():
            parameters = {name: p.ref() for name, p in sl.params.items()}
            block = CircuitBlock.from_layer(type(sl)(**sl.config, **parameters))
            blocks.append(block)
            block_ins = [layers_to_block[sli] for sli in sc.layer_inputs(sl)]
            in_blocks[block] = block_ins
            for bi in block_ins:
                out_blocks[bi].append(block)
            layers_to_block[sl] = block

    # Construct the symbolic circuit obtained by merging multiple circuits
    return Circuit.from_operation(
        scope,
        num_channels,
        blocks,
        in_blocks,
        out_blocks,
        operation=CircuitOperation(operator=CircuitOperator.MERGE, operands=tuple(scs)),
        topologically_ordered=True,
    )


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
    layers_to_block: Dict[Layer, CircuitBlock] = {}

    # For each new circuit block, keep track of (i) its inputs and (ii) the blocks it feeds
    blocks: List[CircuitBlock] = []
    in_blocks: Dict[CircuitBlock, List[CircuitBlock]] = {}
    out_blocks: Dict[CircuitBlock, List[CircuitBlock]] = defaultdict(list)

    for sl in sc.topological_ordering():
        # Input layers get integrated over
        if isinstance(sl, InputLayer) and sl.scope & scope:
            if not (sl.scope <= scope):
                raise NotImplementedError(
                    "Multivariate integration of proper subsets of variables is not implemented"
                )
            # Retrieve the integration rule from the registry and apply it
            func = registry.retrieve_rule(LayerOperation.INTEGRATION, type(sl))
            int_block = func(sl)
            blocks.append(int_block)
            layers_to_block[sl] = int_block
            continue
        assert isinstance(
            sl, (SumLayer, ProductLayer)
        ), "Symbolic inner layers must be either sum or product layers"
        # Sum/product layers are simply copied
        # Note that this willTo keep track of shared parameters, we use parameter references
        parameters = {name: p.ref() for name, p in sl.params.items()}
        int_block = CircuitBlock.from_layer(type(sl)(**sl.config, **parameters))
        blocks.append(int_block)
        layers_to_block[sl] = int_block
        int_block_ins = [layers_to_block[isl] for isl in sc.layer_inputs(sl)]
        in_blocks[int_block] = int_block_ins
        for bi in int_block_ins:
            out_blocks[bi].append(int_block)

    # Construct the integral symbolic circuit and set the integration operation metadata
    return Circuit.from_operation(
        sc.scope,
        sc.num_channels,
        blocks,
        in_blocks,
        out_blocks,
        operation=CircuitOperation(
            operator=CircuitOperator.INTEGRATION,
            operands=(sc,),
            metadata=dict(scope=scope),
        ),
        topologically_ordered=True,
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
    layers_to_block: Dict[Tuple[Layer, Layer], CircuitBlock] = {}

    # For each new circuit block, keep track of (i) its inputs and (ii) the blocks it feeds
    blocks: List[CircuitBlock] = []
    in_blocks: Dict[CircuitBlock, List[CircuitBlock]] = {}
    out_blocks: Dict[CircuitBlock, List[CircuitBlock]] = defaultdict(list)

    # Get the first layers to multiply, from the outputs
    to_multiply = []
    for lhs_out, rhs_out in itertools.product(lhs_sc.outputs, rhs_sc.outputs):
        to_multiply.append((lhs_out, rhs_out))

    # Using a stack in place of recursion for better memory efficiency and debugging
    while to_multiply:
        pair = to_multiply[-1]
        if pair in layers_to_block:
            to_multiply.pop()
            continue
        lhs_layer, rhs_layer = pair
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
        not_yet_multiplied = list(filter(lambda p: p not in layers_to_block, next_to_multiply))
        if len(not_yet_multiplied) > 0:
            to_multiply.extend(not_yet_multiplied)
            continue

        # In case all the input have been multiplied, then construct the product layer
        prod_signature = type(lhs_layer), type(rhs_layer)
        func = registry.retrieve_rule(LayerOperation.MULTIPLICATION, *prod_signature)
        prod_block = func(lhs_layer, rhs_layer)
        blocks.append(prod_block)
        # Make the connections
        prod_block_ins = [layers_to_block[p] for p in next_to_multiply]
        in_blocks[prod_block] = prod_block_ins
        for bi in prod_block_ins:
            out_blocks[bi].append(prod_block)
        layers_to_block[pair] = prod_block
        to_multiply.pop()  # Go up in the recursion

    # Construct the product symbolic circuit
    return Circuit.from_operation(
        lhs_sc.scope | rhs_sc.scope,
        lhs_sc.num_channels,
        blocks,
        in_blocks,
        out_blocks,
        operation=CircuitOperation(
            operator=CircuitOperator.MULTIPLICATION, operands=(lhs_sc, rhs_sc)
        ),
        topologically_ordered=True,
    )


def differentiate(sc: Circuit, registry: Optional[OperatorRegistry] = None) -> Circuit:
    if not sc.is_smooth or not sc.is_decomposable:
        raise StructuralPropertyError(
            "Only smooth and decomposable circuits can be efficiently differentiated."
        )

    # Use the registry in the current context, if not specified otherwise
    if registry is None:
        registry = OPERATOR_REGISTRY.get()
