import heapq
import itertools
from collections.abc import Iterable, Sequence
from numbers import Number
from typing import NamedTuple, TypeVar

import numpy as np

from cirkit.symbolic.circuit import (
    Circuit,
    CircuitBlock,
    CircuitOperation,
    CircuitOperator,
    StructuralPropertyError,
    are_compatible,
)
from cirkit.symbolic.layers import (
    EvidenceLayer,
    InputLayer,
    KroneckerLayer,
    Layer,
    LayerOperator,
    ProductLayer,
    SumLayer,
)
from cirkit.symbolic.parameters import ConstantParameter, Parameter
from cirkit.symbolic.registry import OPERATOR_REGISTRY, OperatorRegistry
from cirkit.utils.scope import Scope


def concatenate(scs: Sequence[Circuit], *, registry: OperatorRegistry | None = None) -> Circuit:
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
    num_channels_s = {sc.num_channels for sc in scs}
    if len(num_channels_s) != 1:
        raise ValueError(
            f"Only circuits with the same number of channels can be concatenated, "
            f"but found a set of number of channels {num_channels_s}"
        )
    num_channels = scs[0].num_channels

    # Mapping the symbolic circuit layers with blocks of circuit layers
    layers_to_block: dict[Layer, CircuitBlock] = {}

    # For each new circuit block, keep track of its inputs
    blocks: list[CircuitBlock] = []
    in_blocks: dict[CircuitBlock, list[CircuitBlock]] = {}
    output_blocks: list[CircuitBlock] = []

    # Simply pass the symbolic layers references
    for sc in scs:
        for sl in sc.topological_ordering():
            block = CircuitBlock.from_layer(sl.copyref())
            blocks.append(block)
            block_ins = [layers_to_block[sli] for sli in sc.layer_inputs(sl)]
            in_blocks[block] = block_ins
            layers_to_block[sl] = block
        output_blocks.extend(layers_to_block[sl] for sl in sc.outputs)

    # Construct the symbolic circuit obtained by merging multiple circuits
    return Circuit.from_operation(
        num_channels,
        blocks,
        in_blocks,
        output_blocks,
        operation=CircuitOperation(operator=CircuitOperator.CONCATENATE, operands=tuple(scs)),
    )


def evidence(
    sc: Circuit,
    obs: dict[int, Number | tuple[Number, ...]],
    *,
    registry: OperatorRegistry | None = None,
) -> Circuit:
    """Observe the value of some variables in a symbolic circuit, and represent the given
    evidence as another symbolic circuit.

    Args:
        sc: The symbolic circuit where some variables have to be observed.
        obs: The observation data, stored as a dictionary mapping variable integer identifiers
            to numbers, i.e., either integer, float or complex values. In the case the
            circuit defines multiple channels per variable, then this is a dictionary mapping
            variable integer identifiers to tuples of as many numbers as the number of channels.
        registry: A registry of symbolic layer operators. If it is None, then the one in
            the current context will be used. See the
            [OPERATOR_REGISTRY][cirkit.symbolic.registry.OPERATOR_REGISTRY] context variable
            for more details.

    Returns:
        The symbolic circuit representing the observation of variables in the given circuit.

    Raises:
        ValueError: If the observation contains variables not defined in the scope of the circuit.
    """
    if not all(
        (isinstance(value, Number) or len(value) == 1)
        if sc.num_channels == 1
        else len(value) == sc.num_channels
        for (var, value) in obs.items()
    ):
        raise ValueError(
            "The observation of each variable should contain as many "
            "values as the number of channels"
        )
    # Check the variables to observe
    scope = Scope(obs.keys())
    if not scope:
        raise ValueError("There are no variables to observe")
    elif not scope <= sc.scope:
        raise ValueError("The variables to observe must be a subset of the scope of the circuit")

    # Mapping the symbolic circuit layers with blocks of circuit layers
    layers_to_block: dict[Layer, CircuitBlock] = {}

    # For each new circuit block, keep track of its inputs
    blocks: list[CircuitBlock] = []
    in_blocks: dict[CircuitBlock, list[CircuitBlock]] = {}

    for sl in sc.topological_ordering():
        # Check if we have to construct the evidence of an input layer
        if isinstance(sl, InputLayer) and sl.scope & scope:
            if not sl.scope <= scope:
                raise NotImplementedError(
                    f"Only complete evidence of multivariate input layers is supported, "
                    f"found {sl.scope} but computing the evidence over {scope}"
                )

            # Build the observation parameter, as a constant tensor that
            # contains assignments to the variables being observed
            # The shape of the observation parameter is (C, D), where C is the
            # number of channels and D is the number of variables the layer
            # depends on
            obs_shape = sc.num_channels, len(sl.scope)
            # obs_ndarray: An array of shape either (D,) or (D, C)
            obs_ndarray = np.array([obs[var] for var in sorted(sl.scope)])
            obs_ndarray = obs_ndarray[None, :] if len(obs_ndarray.shape) == 1 else obs_ndarray.T
            # A constant parameter of shape (C, D), where C can be 1.
            obs_parameter = ConstantParameter(*obs_shape, value=obs_ndarray)

            # Build the evidence layer, with a reference to the input layer
            evi_sl = EvidenceLayer(sl.copyref(), observation=Parameter.from_input(obs_parameter))
            evi_block = CircuitBlock.from_layer(evi_sl)
            blocks.append(evi_block)
            layers_to_block[sl] = evi_block
            continue
        # Sum/product layers and input layers whose scope does not
        # include variables to observe over are simply copied.
        # Note that to keep track of shared parameters, we use parameter references
        evi_block = CircuitBlock.from_layer(sl.copyref())
        blocks.append(evi_block)
        layers_to_block[sl] = evi_block
        in_blocks[evi_block] = [layers_to_block[isl] for isl in sc.layer_inputs(sl)]

    # Construct the sequence of output blocks
    output_blocks = [layers_to_block[sl] for sl in sc.outputs]

    # Construct the evidence symbolic circuit and set the evidence operation metadata
    return Circuit.from_operation(
        sc.num_channels,
        blocks,
        in_blocks,
        output_blocks,
        operation=CircuitOperation(
            operator=CircuitOperator.EVIDENCE,
            operands=(sc,),
            metadata={"scope": scope},
        ),
    )


def integrate(
    sc: Circuit,
    scope: Scope | None = None,
    *,
    registry: OperatorRegistry | None = None,
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
        ValueError: If the scope to integrate over is not a subset of the scope of the circuit,
            or if it is empty.
    """
    # Check for structural properties
    if not sc.is_smooth or not sc.is_decomposable:
        raise StructuralPropertyError(
            "Only smooth and decomposable circuits can be efficiently integrated."
        )

    # Check the variables to integrate
    if scope is None:
        scope = sc.scope
    if not scope:
        raise ValueError("There are no variables to integrate over")
    elif not scope <= sc.scope:
        raise ValueError(
            "The variables scope to integrate must be a subset of the scope of the circuit"
        )

    # Use the registry in the current context, if not specified otherwise
    if registry is None:
        registry = OPERATOR_REGISTRY.get()

    # Mapping the symbolic circuit layers with blocks of circuit layers
    layers_to_block: dict[Layer, CircuitBlock] = {}

    # For each new circuit block, keep track of its inputs
    blocks: list[CircuitBlock] = []
    in_blocks: dict[CircuitBlock, list[CircuitBlock]] = {}

    for sl in sc.topological_ordering():
        # Input layers get integrated over
        if isinstance(sl, InputLayer) and sl.scope & scope:
            func = registry.retrieve_rule(LayerOperator.INTEGRATION, type(sl))
            int_block = func(sl, scope=scope)
            blocks.append(int_block)
            layers_to_block[sl] = int_block
            continue
        # Sum/product layers and input layers whose scope does not
        # include variables to integrate over are simply passed through
        int_block = CircuitBlock.from_layer(sl.copyref())
        blocks.append(int_block)
        layers_to_block[sl] = int_block
        in_blocks[int_block] = [layers_to_block[isl] for isl in sc.layer_inputs(sl)]

    # Construct the sequence of output blocks
    output_blocks = [layers_to_block[sl] for sl in sc.outputs]

    # Construct the integral symbolic circuit and set the integration operation metadata
    return Circuit.from_operation(
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


def multiply(sc1: Circuit, sc2: Circuit, *, registry: OperatorRegistry | None = None) -> Circuit:
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
    if not are_compatible(sc1, sc2):
        raise StructuralPropertyError(
            "Only compatible circuits can be multiplied into decomposable circuits."
        )

    # Use the registry in the current context, if not specified otherwise
    if registry is None:
        registry = OPERATOR_REGISTRY.get()

    # Map from pairs of layers to their product circuit block
    layers_to_block: dict[tuple[Layer, Layer], CircuitBlock] = {}

    # For each new circuit block, keep track of its inputs
    blocks: list[CircuitBlock] = []
    in_blocks: dict[CircuitBlock, list[CircuitBlock]] = {}

    # Get the first layers to multiply, from the outputs
    to_multiply = []
    for l1, l2 in itertools.product(sc1.outputs, sc2.outputs):
        to_multiply.append((l1, l2))

    # Using a stack in place of recursion for better memory efficiency and debugging
    while to_multiply:
        pair = to_multiply[-1]

        # Check if the layers have been already multiplied
        if pair in layers_to_block:
            to_multiply.pop()
            continue

        # Get the layers to multiply
        l1, l2 = pair

        # Check whether we are multiplying layers over disjoint scope
        # If that is the case, then we just need to introduce a Kronecker product layer
        if len(sc1.layer_scope(l1) & sc2.layer_scope(l2)) == 0:
            if l1.num_output_units != l2.num_output_units:
                raise NotImplementedError(
                    f"Layers over disjoint scopes can be multiplied if they have the same size, "
                    f"but found layer sizes {l1.num_output_units} and {l2.num_output_units}"
                )
            # Get the sub-circuits rooted in the layers being multiplied
            sub1_circuit = sc1.subgraph(l1)
            sub2_circuit = sc2.subgraph(l2)
            # Copy the layers and the connections, by making references to the parameters
            sub1_blocks = {l: CircuitBlock.from_layer(l.copyref()) for l in sub1_circuit.layers}
            sub2_blocks = {l: CircuitBlock.from_layer(l.copyref()) for l in sub2_circuit.layers}
            blocks.extend(sub1_blocks.values())
            blocks.extend(sub2_blocks.values())
            in_blocks.update(
                (b, [sub1_blocks[li] for li in sc1.layer_inputs(l)]) for l, b in sub1_blocks.items()
            )
            in_blocks.update(
                (b, [sub2_blocks[li] for li in sc2.layer_inputs(l)]) for l, b in sub2_blocks.items()
            )
            # Introduce a fresh kronecker product layer, which will multiply
            # the two layers over disjoint scope
            kl = CircuitBlock.from_layer(KroneckerLayer(l1.num_output_units, arity=2))
            blocks.append(kl)
            in_blocks[kl] = [sub1_blocks[l1], sub2_blocks[l2]]
            layers_to_block[pair] = kl
            # Go up in the recursion
            to_multiply.pop()
            continue

        # We need to multiply layers over overlapping scopes
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
            l1_inputs = sorted(l1_inputs, key=sc1.layer_scope)
            l2_inputs = sorted(l2_inputs, key=sc2.layer_scope)
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
        # Go up in the recursion
        to_multiply.pop()

    # Construct the sequence of output blocks
    output_blocks = [
        layers_to_block[(l1, l2)] for l1, l2 in itertools.product(sc1.outputs, sc2.outputs)
    ]

    # Construct the product symbolic circuit
    return Circuit.from_operation(
        sc1.num_channels,
        blocks,
        in_blocks,
        output_blocks,
        operation=CircuitOperation(operator=CircuitOperator.MULTIPLICATION, operands=(sc1, sc2)),
    )


class _ScopeVarAndBlockAndInputs(NamedTuple):
    """The tuple of a scope variable and a circiut block for diff.

    Used for differential of ProductLayer.
    """

    scope_var: int  # The id of a variable in the scope of THE ProductLayer.
    diff_block: CircuitBlock  # The partial diff of THE ProductLayer w.r.t. the var.
    diff_in_blocks: list[CircuitBlock]  # The inputs to the layer of diff_block.


_T = TypeVar("_T")  # TODO: for _repeat. move together


# TODO: this can be made public and moved to utils, might be used elsewhere.
def _repeat(iterable: Iterable[_T], /, *, times: int) -> Iterable[_T]:
    """Repeat each element of the given iterable by given times.

    The elements are generated lazily. The iterable passed in will be iterated once.
    This function differs from itertools in that it repeats an interable instead of only one elem.

    Args:
        iterable (Iterable[_T]): The iterable to generate the original elements.
        times (int): The times to repeat each element.

    Returns:
        Iterable[_T]: The iterable with repeated elements.
    """
    return itertools.chain.from_iterable(itertools.repeat(elem, times=times) for elem in iterable)


def differentiate(
    sc: Circuit, registry: OperatorRegistry | None = None, *, order: int = 1
) -> Circuit:
    if not sc.is_smooth or not sc.is_decomposable:
        raise StructuralPropertyError(
            "Only smooth and decomposable circuits can be efficiently differentiated."
        )
    if order <= 0:
        raise ValueError("The order of differentiation must be positive.")

    # Use the registry in the current context, if not specified otherwise
    if registry is None:
        registry = OPERATOR_REGISTRY.get()

    # Mapping the symbolic circuit layers with blocks of circuit layers
    layers_to_blocks: dict[Layer, list[CircuitBlock]] = {}

    # For each new circuit block, keep track of its inputs
    in_blocks: dict[CircuitBlock, Sequence[CircuitBlock]] = {}

    for sl in sc.topological_ordering():
        # "diff_blocks: List[CircuitBlock]" is the diff of sl wrt each variable and channel in order
        #                                   and then at the end we append a copy of sl

        if isinstance(sl, InputLayer):
            # TODO: no type hint for func, also cannot quick jump in static analysis
            func = registry.retrieve_rule(LayerOperator.DIFFERENTIATION, type(sl))
            diff_blocks = [
                func(sl, var_idx=var_idx, ch_idx=ch_idx, order=order)
                for var_idx, ch_idx in itertools.product(
                    range(len(sl.scope)), range(sc.num_channels)
                )
            ]

        elif isinstance(sl, SumLayer):
            # Zip to transpose the generator into an iterable of length (num_vars * num_chs),
            #   corresponding to each var to take diff.
            # Each item is a tuple of length arity, which are inputs to that diff.
            # TODO: typeshed issue?
            # ANNOTATE: zip gives Any when using *iterables.
            zip_blocks_in: Iterable[tuple[CircuitBlock, ...]] = zip(
                # This is a generator of length arity, corresponding to each input of sl.
                # Each item is a list of length (num_vars * num_chs), corresponding to the diff wrt
                #   each variable of that input.
                # NOTE: [-1] is omitted and will be added at the end.
                *(layers_to_blocks[sl_in][:-1] for sl_in in sc.layer_inputs(sl))
            )

            # The layers are the same for all diffs of a SumLayer. We retrieve (num_vars * num_chs)
            #   from the length of one input blocks.
            var_ch = len(layers_to_blocks[sc.layer_inputs(sl)[0]][:-1])
            diff_blocks = [CircuitBlock.from_layer(sl.copyref()) for _ in range(var_ch)]

            # Connect the layers to their inputs, by zipping a length of (num_vars * num_chs).
            in_blocks.update(zip(diff_blocks, zip_blocks_in))

        elif isinstance(sl, ProductLayer):
            # NOTE: Only the outmost level can be a generator, and inner levels must be lists,
            #       otherwise reference to locals will be broken.

            # This is a generator of length arity, corresponding to each input of sl.
            # Each item is a list of length (num_vars * num_chs) of that input, corresponding to the
            #   diff wrt each var and ch of that input.
            all_scope_var_diff_block = (
                # Each list is all the diffs of sl wrt each var and each channel in the scope of
                #   the cur_layer in the input of sl.
                [
                    # Each named-tuple is a diff of sl and its inputs, where the diff is wrt the
                    #   current variable and channel as in the double loop.
                    _ScopeVarAndBlockAndInputs(
                        # Label the named-tuple as the var id in the whole scope, for sorting.
                        scope_var=scope_var,
                        # The layers are the same for all diffs of a ProductLayer.
                        diff_block=CircuitBlock.from_layer(sl.copyref()),
                        # The inputs to the diff is the copy of input to sl (retrieved by [-1]),
                        #   only with cur_layer replaced by its diff.
                        diff_in_blocks=[
                            diff_cur_layer if sl_in == cur_layer else layers_to_blocks[sl_in][-1]
                            for sl_in in sc.layer_inputs(sl)
                        ],
                    )
                    # Loop over the (num_vars * num_chs) diffs of cur_layer, while also providing
                    #   the corresponding scope_var which the current diff is wrt.
                    # We need the scope_var to label and sort the diff layers of sl. We do nnt need
                    #   channel ids because they are always saved densely in order.
                    for scope_var, diff_cur_layer in zip(
                        _repeat(sc.layer_scope(cur_layer), times=sc.num_channels),
                        layers_to_blocks[cur_layer][:-1],
                    )
                ]
                # Loop over each input of sl for the diffs wrt vars and chs in its scope.
                for cur_layer in sc.layer_inputs(sl)
            )

            # NOTE: This relys on the fact that Scope object is iterated in id order.
            # Merge sort the named-tuples by the var id in the scope, so that the diffs are
            #   correctly ordered according to the scope of sl.
            sorted_scope_var_diff_block = list(
                heapq.merge(
                    # Unpack the generator into several lists, where each list is the named-tuples
                    #   wrt the scope of each input to sl.
                    *all_scope_var_diff_block,
                    key=lambda scope_var_diff_block: scope_var_diff_block.scope_var,
                )
            )

            # Take out the diffs of sl and save them in diff_blocks in correct order.
            diff_blocks = [
                scope_var_diff_block.diff_block
                for scope_var_diff_block in sorted_scope_var_diff_block
            ]

            # Connect the diffs with its corresponding inputs as saved in the named-tuples.
            in_blocks.update(
                (scope_var_diff_block.diff_block, scope_var_diff_block.diff_in_blocks)
                for scope_var_diff_block in sorted_scope_var_diff_block
            )

        else:
            # NOTE: In the above if/elif, we made all conditions explicit to make it more readable
            #       and also easier for static analysis inside the blocks. Yet the completeness
            #       cannot be inferred and is only guaranteed by larger picture. Also, should
            #       anything really go wrong, we will hit this guard statement instead of going into
            #       a wrong branch.
            assert False, "This should not happen."

        # Save sl in the diff circuit and connect inputs. This can be accessed through
        #   diff_blocks[-1], as in the [-1] above for ProductLayer.
        diff_blocks.append(CircuitBlock.from_layer(sl.copyref()))
        in_blocks[diff_blocks[-1]] = [layers_to_blocks[sl_in][-1] for sl_in in sc.layer_inputs(sl)]

        # Save all the blocks including a copy of sl at [-1] as the diff layers of sl.
        layers_to_blocks[sl] = diff_blocks

    # Construct the integral symbolic circuit and set the integration operation metadata
    return Circuit.from_operation(
        sc.num_channels,
        sum(layers_to_blocks.values(), []),
        in_blocks,
        sum((layers_to_blocks[sl] for sl in sc.outputs), []),
        operation=CircuitOperation(
            operator=CircuitOperator.DIFFERENTIATION,
            operands=(sc,),
            metadata=dict(order=order),
        ),
    )


def conjugate(
    sc: Circuit,
    *,
    registry: OperatorRegistry | None = None,
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
    layers_to_block: dict[Layer, CircuitBlock] = {}

    # For each new circuit block, keep track of its inputs
    blocks: list[CircuitBlock] = []
    in_blocks: dict[CircuitBlock, list[CircuitBlock]] = {}

    for sl in sc.topological_ordering():
        # The conjugation of a product layer is equivalent to the product of its conjugated inputs
        if isinstance(sl, ProductLayer):
            conj_block = CircuitBlock.from_layer(sl)
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
        sc.num_channels,
        blocks,
        in_blocks,
        output_blocks,
        operation=CircuitOperation(operator=CircuitOperator.CONJUGATION, operands=(sc,)),
    )
