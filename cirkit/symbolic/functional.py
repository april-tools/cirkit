import heapq
import itertools
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Iterable, NamedTuple, TypeVar

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
from cirkit.symbolic.parameters import (
    ConstantParameter,
    GateFunctionParameter,
    Parameter,
    TensorParameter,
)
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
    """
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
            to numbers, i.e., either integer, float or complex values.
        registry: A registry of symbolic layer operators. If it is None, then the one in
            the current context will be used. See the
            [OPERATOR_REGISTRY][cirkit.symbolic.registry.OPERATOR_REGISTRY] context variable
            for more details.

    Returns:
        The symbolic circuit representing the observation of variables in the given circuit.

    Raises:
        ValueError: If the observation contains variables not defined in the scope of the circuit.
        NotImplementedError: If the evidence of a multivariate input layer needs to be constructed.
    """
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
            # The shape of the observation parameter is (D,), where D
            # is the number of variables the layer depends on
            obs_ndarray = np.array([obs[var] for var in sorted(sl.scope)])
            # A constant parameter of shape (D,)
            obs_parameter = ConstantParameter(len(sl.scope), value=obs_ndarray)

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
    r"""Integrate the function computed by a circuit, and represent it as another circuit.
    This operator requires the given circuit to be both smooth and decomposable.

    For example, integrate can be used to compute the partition function in
    probabilistic circuits, e.g., in
    [the Sum-of-Squares notebook](https://github.com/april-tools/cirkit/blob/main/notebooks/sum-of-squares-circuits.ipynb).

    Formally, given a symbolic circuit $c$ over a set of variables $\mathbf{X}$, integrate
    over a scope $\mathbf{Z}\subseteq\mathbf{X}$ returns another symbolic circuit $c'$ such that
    $$
    c'(\mathbf{Y}) = \int_{\mathbf{z}\in\mathrm{dom}(\mathbf{Z})} c(\mathbf{Y},\mathbf{z}) \mathrm{d}\mathbf{Z},
    $$
    where $\mathbf{Y} = \mathbf{X}\setminus\mathbf{Z}$ is the set of remaining variables.

    If you want to integrate an already-compiled circuit without creating a new symbolic
    circuit, then have a look at [IntegrateQuery][cirkit.backend.torch.queries].

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
    if not scope <= sc.scope:
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
    r"""Multiply two symbolic circuits and represent the result as another circuit.
    This operator requires the input circuits to be
    [smooth][cirkit.symbolic.circuit.Circuit.is_smooth],
    [decomposable][cirkit.symbolic.circuit.Circuit.is_decomposable],
    and [compatible][cirkit.symbolic.circuit.are_compatible].
    The resulting circuit will be smooth and decomposable.
    Moreover, if the input circuits are structured decomposable,
    then the resulting circuit will also be structured decomposable,
    and compatible with the inputs.

    The product of circuits can be used to compute expectations (by composing the multiply and
    [integrate][cirkit.symbolic.functional.integrate] operators), or to build squared probabilistic
    circuits, as in
    [the Sum-of-Squares notebook](https://github.com/april-tools/cirkit/blob/main/notebooks/sum-of-squares-circuits.ipynb).

    Formally, given two compatible circuits $c_1$, $c_2$ having the same variables scope
    $\mathbf{X}$, multiply returns another circuit $c'$ such that it encodes
    $c'(\mathbf{X}) = c_1(\mathbf{X})\cdot c_2(\mathbf{X})$.

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
        if not sc1.layer_scope(l1) & sc2.layer_scope(l2):
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


def differentiate(
    sc: Circuit, order: int = 1, *, registry: OperatorRegistry | None = None
) -> Circuit:
    """Represent the differential of a symbolic circuit with respect to its variables scope
        as a circuit. The operator requires the input circuit to be smooth and decomposable,
        and a higher-order differential can be computed. The symbolic circuit resulting from the
        differentiation operator is another smooth and decomposable circuit with as many output
        layers as the number of variables in the scope of the input circuit.

    Args:
        sc: The symbolic circuit.
        order: The differentiation order.
        registry: A registry of symbolic layer operators. If it is None, then the one in
            the current context will be used. See the
            [OPERATOR_REGISTRY][cirkit.symbolic.registry.OPERATOR_REGISTRY] context variable
            for more details.

    Returns:
        A multi-output smooth and decomposable symbolic circuit computing the
            differential of the input circuit with respect to each variable.

    Raises:
        StructuralPropertyError: If the given circuit is not smooth and decomposable.
        ValueError: If the given differentiation order is not a positive integer.
    """
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
        # "diff_blocks: List[CircuitBlock]" is the diff of sl wrt each variable in order
        #                                   and then at the end we append a copy of sl

        if isinstance(sl, InputLayer):
            # TODO: no type hint for func, also cannot quick jump in static analysis
            func = registry.retrieve_rule(LayerOperator.DIFFERENTIATION, type(sl))
            diff_blocks = [
                func(sl, var_idx=var_idx, order=order) for var_idx in range(len(sl.scope))
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
                # Each list is all the diffs of sl wrt each var in the scope of
                #   the cur_layer in the input of sl.
                [
                    # Each named-tuple is a diff of sl and its inputs, where the diff is wrt the
                    #   current variable as in the double loop.
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
                    # We need the scope_var to label and sort the diff layers of sl.
                    for scope_var, diff_cur_layer in zip(
                        sc.layer_scope(cur_layer),
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
        sum(layers_to_blocks.values(), []),
        in_blocks,
        sum((layers_to_blocks[sl] for sl in sc.outputs), []),
        operation=CircuitOperation(
            operator=CircuitOperator.DIFFERENTIATION,
            operands=(sc,),
            metadata={"order": order},
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
        blocks,
        in_blocks,
        output_blocks,
        operation=CircuitOperation(operator=CircuitOperator.CONJUGATION, operands=(sc,)),
    )


GateFunctionSpecs = dict[str, list[Layer]]
"""The gate function specification. It is a map from an id (a string) to the list of layers 
it should parametrize."""


GateFunctionParameterSpecs = dict[str, tuple[int, ...]]
"""The gate function parameter specification.
It is a map from a gate function name to a parameter group specification.
The parameter group specification is the shapes that must be computed for that parameter.
Groups are constructed by collating together parameters with the same shape."""


def model_parameterize(
    circuit: Circuit, *, gate_functions: GateFunctionSpecs
) -> tuple[Circuit, GateFunctionParameterSpecs]:
    """Parameterize some layers of a symbolic circuit by means of an externally-provided model.

    Args:
        circuit: The symbolic circuit.
        gate_functions: A mapping from a gate function identifier to the
            list of layers it will parametrize.

    Returns:
        tuple[Circuit, GateFunctionParameterSpecs]: A pair where the first element is a new symbolic
            circuit whose tensor parameters have been substituted by the symbolic information that
            the value of those parameters will come from an externally defined model. The second
            element is a map from a gate function name to a parameter group specification.
            The parameter group specification is the shapes that must be computed for that parameter.
            Groups are constructed by collating together parameters with the same shape..

    Raises:
        ValueError: If the given circuit is the output of a circuit operator.
            That is, this function is designed for circuits that are entry points to a circuit
            pipeline.
        ValueError: If the provided gate functions are not defined on pairwise mutually disjoint
            sets of layers.
    """
    if circuit.operation is not None:
        raise ValueError("The circuit to parameterize must not be the output of a circuit operator")

    # the layers specified in the gate function specification all be mutually disjoint
    if any(
        len(set(l1).intersection(l2)) != 0
        for l1, l2 in itertools.combinations(gate_functions.values(), 2)
    ):
        raise ValueError("The gate functions must parametrize mutually disjoint set of layers.")

    # group all parameters from the same gate function together so that they can be computed
    # in folded fashion and upacked later
    # make sure that all elements within a group are compatible (same parameter type and shape)
    gate_function_specs: GateFunctionParameterSpecs = {}
    layers_map: dict[Layer, Layer] = {l: l.copy() for l in circuit.layers}

    for gf_group, gf_layers in gate_functions.items():
        # group layers together based on metadata
        layers_metadata = defaultdict(list)
        for l in gf_layers:
            for p_k, p in l.params.items():
                layers_metadata[(p_k, p.shape)].append(l)

        # construct the gate function spec for each group
        # since parameters with same name from same group might have different
        # shapes we group all the ones with same shape under a common name
        for g_i, ((p_name, p_shape), g_layers) in enumerate(layers_metadata.items()):
            gf_name = f"{gf_group}.{p_name}.{g_i}"

            # the gate function can compute |g_elements| parameters at once
            # all of shape g_shape
            gate_function_specs[gf_name] = (len(g_layers), *p_shape)

            # Replace the parameter tensor to be externally parameterized by a gate function
            for i_sl, sl in enumerate(g_layers):
                # replace layer parameter with gate function
                gf = GateFunctionParameter(*p_shape, name=gf_name, index=i_sl)
                layers_map[sl] = sl.copy(params={p_name: Parameter.from_input(gf)})

    # Construct the resulting circuit
    # use a shallow copy of the parameters that have not been changed
    layers = [layers_map[l] for l in circuit.layers]
    in_layers = {
        sl: [layers_map[prev_sli] for prev_sli in circuit.layer_inputs(prev_sl)]
        for prev_sl, sl in layers_map.items()
    }
    output_layers = [layers_map[prev_sli] for prev_sli in circuit.outputs]
    circuit = Circuit(layers, in_layers=in_layers, outputs=output_layers)

    # Return both the resulting circuit and the model parameter specifications
    return circuit, gate_function_specs
