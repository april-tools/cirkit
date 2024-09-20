import itertools
from typing import Dict, List, Optional, Sequence, Tuple

from cirkit.symbolic.circuit import (
    Circuit,
    CircuitBlock,
    CircuitOperation,
    CircuitOperator,
    StructuralPropertyError,
    are_compatible,
)
from cirkit.symbolic.layers import InputLayer, Layer, LayerOperator, ProductLayer, SumLayer
from cirkit.symbolic.registry import OPERATOR_REGISTRY, OperatorRegistry
from cirkit.utils.scope import Scope


def concatenate(scs: Sequence[Circuit], registry: Optional[OperatorRegistry] = None) -> Circuit:
    """Concatenates a sequence of symbolic circuits. Concatenating circuits means constructing
    another circuit such that its output layers consists of the output layers of each circuit
    (in the given order). This operator does not require the satisfaction of any structural
    property by the given circuits.

    Args:
        scs: A sequence of symbolic circuits.
        registry: A registry of symbolic layer operators. It is not used for this operator.

    Returns:
        A circuit obtained by concatenating circuits.

    Raises:
        ValueError: If the given circuits to concatenate have different number of channels per
            variable.
    """
    # Retrieve the number of channels
    num_channels_s = set(sc.num_channels for sc in scs)
    if len(num_channels_s) != 1:
        raise ValueError(
            f"Only circuits with the same number of channels can be concatenated, "
            f"but found a set of number of channels {num_channels_s}"
        )
    num_channels = scs[0].num_channels

    # Mapping the symbolic circuit layers with blocks of circuit layers
    layers_to_block: Dict[Layer, CircuitBlock] = {}

    # For each new circuit block, keep track of its inputs
    blocks: List[CircuitBlock] = []
    in_blocks: Dict[CircuitBlock, List[CircuitBlock]] = {}
    output_blocks: List[CircuitBlock] = []

    # Copy the symbolic layers, pick references to parameters and build the blocks
    for sc in scs:
        for sl in sc.topological_ordering():
            parameters = {name: p.ref() for name, p in sl.params.items()}
            block = CircuitBlock.from_layer(type(sl)(**sl.config, **parameters))
            blocks.append(block)
            block_ins = [layers_to_block[sli] for sli in sc.layer_inputs(sl)]
            in_blocks[block] = block_ins
            layers_to_block[sl] = block
        output_blocks.extend(layers_to_block[sl] for sl in sc.outputs)

    # Retrieve the union of the scopes of the circuits
    scope = Scope.union(*tuple(sc.scope for sc in scs))

    # Construct the symbolic circuit obtained by merging multiple circuits
    return Circuit.from_operation(
        scope,
        num_channels,
        blocks,
        in_blocks,
        output_blocks,
        operation=CircuitOperation(operator=CircuitOperator.CONCATENATE, operands=tuple(scs)),
    )


def integrate(
    sc: Circuit,
    scope: Optional[Scope] = None,
    registry: Optional[OperatorRegistry] = None,
) -> Circuit:
    """Integrate the function computed by a circuit, and represent it as another circuit.
    This operator requires the given circuit to be both smooth and decomposable.

    Args:
        sc: A symbolic circuit.
        scope: The varaibles scope to integrate over. If it is None, then all variables on
            which the given circuit is defined on will be integrated over.
        registry: A registry of symbolic layer operators. If it is None, then the one in
            the current context will be used. See the
            [OPERATOR_REGISTRY][cirkit.symbolic.registry.OPERATOR_REGISTRY] context variable
            for more details.

    Returns:
        The symbolic circuit reprenting the integration operation of the given circuit.

    Raises:
        StructuralPropertyError: If the given circuit is not smooth and decomposable.
        ValueError: If the scope to integrate over is not a subset of the scope of the circuit.
    """
    # Check for structural properties
    if not sc.is_smooth or not sc.is_decomposable:
        raise StructuralPropertyError(
            "Only smooth and decomposable circuits can be efficiently integrated."
        )

    # Check the variable
    if scope is None:
        scope = sc.scope
    elif not scope <= sc.scope:
        raise ValueError(
            "The variables scope to integrate must be a subset of the scope of the circuit"
        )

    # Use the registry in the current context, if not specified otherwise
    if registry is None:
        registry = OPERATOR_REGISTRY.get()

    # Mapping the symbolic circuit layers with blocks of circuit layers
    layers_to_block: Dict[Layer, CircuitBlock] = {}

    # For each new circuit block, keep track of its inputs
    blocks: List[CircuitBlock] = []
    in_blocks: Dict[CircuitBlock, List[CircuitBlock]] = {}

    for sl in sc.topological_ordering():
        # Input layers get integrated over
        if isinstance(sl, InputLayer) and sl.scope & scope:
            func = registry.retrieve_rule(LayerOperator.INTEGRATION, type(sl))
            int_block = func(sl, scope=scope)
            blocks.append(int_block)
            layers_to_block[sl] = int_block
            continue
        # Sum/product layers and input layers whose scope does not
        # include variables to integrate over are simply copied.
        # Note that this willTo keep track of shared parameters, we use parameter references
        parameters = {name: p.ref() for name, p in sl.params.items()}
        int_block = CircuitBlock.from_layer(type(sl)(**sl.config, **parameters))
        blocks.append(int_block)
        layers_to_block[sl] = int_block
        in_blocks[int_block] = [layers_to_block[isl] for isl in sc.layer_inputs(sl)]

    # Construct the sequence of output blocks
    output_blocks = [layers_to_block[sl] for sl in sc.outputs]

    # Construct the integral symbolic circuit and set the integration operation metadata
    return Circuit.from_operation(
        sc.scope,
        sc.num_channels,
        blocks,
        in_blocks,
        output_blocks,
        operation=CircuitOperation(
            operator=CircuitOperator.INTEGRATION,
            operands=(sc,),
            metadata={"scope": scope},
        ),
    )


def multiply(sc1: Circuit, sc2: Circuit, registry: Optional[OperatorRegistry] = None) -> Circuit:
    """Multiply two symbolic circuit and represent it as another circuit.
    This operator requires the input circuits to be smooth, decomposable and compatible.
    The resulting circuit will be smooth and decomposable. Moreover, if the input circuits
    are structured decomposable, then the resulting circuit will also be structured.

    Args:
        sc1: The first symbolic circuit.
        sc2: The second symbolic circuit.
        registry: A registry of symbolic layer operators. If it is None, then the one in
            the current context will be used. See the
            [OPERATOR_REGISTRY][cirkit.symbolic.registry.OPERATOR_REGISTRY] context variable
            for more details.

    Returns:
        The symbolic circuit representing the multiplication of the given circuits.

    Raises:
        NotImplementedError: If the given circuits have different scope.
        StructuralPropertyError: If the given circuits are not smooth and decomposable,
            or if they are not compatible with each other.
    """
    if sc1.scope != sc2.scope:
        raise NotImplementedError("Only the product of circuits over the same scope is implemented")
    scope = sc1.scope
    if not are_compatible(sc1, sc2):
        raise StructuralPropertyError(
            "Only compatible circuits can be multiplied into decomposable circuits."
        )

    # Use the registry in the current context, if not specified otherwise
    if registry is None:
        registry = OPERATOR_REGISTRY.get()

    # Map from pairs of layers to their product circuit block
    layers_to_block: Dict[Tuple[Layer, Layer], CircuitBlock] = {}

    # For each new circuit block, keep track of its inputs
    blocks: List[CircuitBlock] = []
    in_blocks: Dict[CircuitBlock, List[CircuitBlock]] = {}

    # Get the first layers to multiply, from the outputs
    to_multiply = []
    for l1, l2 in itertools.product(sc1.outputs, sc2.outputs):
        to_multiply.append((l1, l2))

    # Using a stack in place of recursion for better memory efficiency and debugging
    while to_multiply:
        pair = to_multiply[-1]
        if pair in layers_to_block:
            to_multiply.pop()
            continue
        l1, l2 = pair
        l1_inputs = sc1.layer_inputs(l1)
        l2_inputs = sc2.layer_inputs(l2)
        if isinstance(l1, InputLayer):
            # TODO: generalize product between input and inner layers
            next_to_multiply = []
        elif isinstance(l1, SumLayer):
            # TODO: generalize product between input and inner layers
            next_to_multiply = list(itertools.product(l1_inputs, l2_inputs))
        elif isinstance(l1, ProductLayer):
            # TODO: generalize product such that it can multiply layers of different arity
            #       this is related to the much more relaxed definition of compatibility between
            #       circuits
            assert len(l1_inputs) == len(l2_inputs)
            # Sort layers based on the scope, such that we can multiply layers with matching scopes
            l1_inputs = sorted(l1_inputs, key=lambda sl: sl.scope)
            l2_inputs = sorted(l2_inputs, key=lambda sl: sl.scope)
            next_to_multiply = list(zip(l1_inputs, l2_inputs))
        else:
            assert False

        # Check if at least one pair of layers needs to be multiplied before going up in the
        # recursion
        not_yet_multiplied = [p for p in next_to_multiply if p not in layers_to_block]
        if len(not_yet_multiplied) > 0:
            to_multiply.extend(not_yet_multiplied)
            continue

        # In case all the input have been multiplied, then construct the product layer
        prod_signature = type(l1), type(l2)
        func = registry.retrieve_rule(LayerOperator.MULTIPLICATION, *prod_signature)
        prod_block = func(l1, l2)
        blocks.append(prod_block)
        # Make the connections
        in_blocks[prod_block] = [layers_to_block[p] for p in next_to_multiply]
        layers_to_block[pair] = prod_block
        to_multiply.pop()  # Go up in the recursion

    # Construct the sequence of output blocks
    output_blocks = [
        layers_to_block[(l1, l2)] for l1, l2 in itertools.product(sc1.outputs, sc2.outputs)
    ]

    # Construct the product symbolic circuit
    return Circuit.from_operation(
        scope,
        sc1.num_channels,
        blocks,
        in_blocks,
        output_blocks,
        operation=CircuitOperation(operator=CircuitOperator.MULTIPLICATION, operands=(sc1, sc2)),
    )


def differentiate(sc: Circuit, registry: Optional[OperatorRegistry] = None) -> Circuit:
    if not sc.is_smooth or not sc.is_decomposable:
        raise StructuralPropertyError(
            "Only smooth and decomposable circuits can be efficiently differentiated."
        )

    # Use the registry in the current context, if not specified otherwise
    if registry is None:
        registry = OPERATOR_REGISTRY.get()
    raise NotImplementedError()


def conjugate(
    sc: Circuit,
    registry: Optional[OperatorRegistry] = None,
) -> Circuit:
    """Apply the complex conjugation operator to a symbolic circuit, and represent it as another
    circuit. This operator does not require the satisfaction of structural properties. Moreover,
    the resulting circuit will inherit the structural property of the given circuit.

    Args:
        sc: A symbolic circuit.
        registry: A registry of symbolic layer operators. If it is None, then the one in
            the current context will be used. See the
            [OPERATOR_REGISTRY][cirkit.symbolic.registry.OPERATOR_REGISTRY] context variable
            for more details.

    Returns:
        The symbolic circuit representing the complex conjugation operation of the given circuit.
    """
    # Use the registry in the current context, if not specified otherwise
    if registry is None:
        registry = OPERATOR_REGISTRY.get()

    # Mapping the symbolic circuit layers with blocks of circuit layers
    layers_to_block: Dict[Layer, CircuitBlock] = {}

    # For each new circuit block, keep track of its inputs
    blocks: List[CircuitBlock] = []
    in_blocks: Dict[CircuitBlock, List[CircuitBlock]] = {}

    for sl in sc.topological_ordering():
        # The conjugation of a product layer is equivalent to the product of its conjugated inputs
        if isinstance(sl, ProductLayer):
            parameters = {name: p.ref() for name, p in sl.params.items()}
            conj_block = CircuitBlock.from_layer(type(sl)(**sl.config, **parameters))
            blocks.append(conj_block)
            layers_to_block[sl] = conj_block
            in_blocks[conj_block] = [layers_to_block[isl] for isl in sc.layer_inputs(sl)]
            continue

        # We are not taking the conjugation of a non-product layer
        # Retrieve the conjugation rule from the registry and apply it
        assert isinstance(sl, (InputLayer, SumLayer))
        func = registry.retrieve_rule(LayerOperator.CONJUGATION, type(sl))
        conj_block = func(sl)
        blocks.append(conj_block)
        layers_to_block[sl] = conj_block
        in_blocks[conj_block] = [layers_to_block[isl] for isl in sc.layer_inputs(sl)]

    # Construct the sequence of output blocks
    output_blocks = [layers_to_block[sl] for sl in sc.outputs]

    # Construct the conjugate symbolic circuit
    return Circuit.from_operation(
        sc.scope,
        sc.num_channels,
        blocks,
        in_blocks,
        output_blocks,
        operation=CircuitOperation(operator=CircuitOperator.CONJUGATION, operands=(sc,)),
    )
