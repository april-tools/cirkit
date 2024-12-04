from collections import defaultdict
from collections.abc import Iterable

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import Layer
from cirkit.symbolic.parameters import FunctionParameter, Parameter, TensorParameter

FunctionParameterSpecs = dict[str, tuple[int, ...]]


def parameterize(
    circuit: Circuit,
    *,
    function: str,
    filter_layers: list[type[Layer]],
) -> tuple[Circuit, FunctionParameterSpecs]:
    """Parameterize some layers of a symbolic circuit by means of an externally-provided function.

    Args:
        circuit: The symbolic circuit.
        function: The function identifier to use.
        filter_layers: The list of symbolic layer types to parameterize.

    Returns:
        tuple[Circuit, FunctionParameterSpecs]: A pair where the first element is a new symbolic
            circuit whose tensor parameters have been substituted by the symbolic information that
            the value of those parameters will come from an externally defined function. The second
            element is a dictionary mapping coalesced tensor parameter names to their shape.

    Raises:
        ValueError: If the given circuit is the output of a circuit operator.
            That is, this function is designed for circuits that are entry points to a circuit
            pipeline.
    """
    if circuit.operation is not None:
        raise ValueError("The circuit to parameterize must not be the output of a circuit operator")

    def _filter_layers(sls: Iterable[Layer]) -> tuple[list[Layer], list[tuple[Layer, list[str]]]]:
        # Retrieve the layers to externally parameterize, together with the list of identifiers
        # of the parameters to be externally parameterized
        rest_of_layers: list[Layer] = []
        layers_to_parameterize: list[tuple[Layer, list[str]]] = []
        for sl in sls:
            # First, we filter-out those layers that have not been specified
            if not any(isinstance(sl, sl_cls) for sl_cls in filter_layers):
                rest_of_layers.append(sl)
                continue
            # Second, we yield the layers that have at least one parameter computational graph
            # consisting of a single tensor parameter. E.g., we drop a sum layer having
            # a softmax re-parameterization, but we retain it if it has no re-parameterization
            # In addition, we return the parameter name identifiers
            pnames: list[str] = []
            for pname, pgraph in sl.params.items():
                if len(pgraph.nodes) > 1:
                    continue
                in_node = pgraph.nodes[0]
                if not isinstance(in_node, TensorParameter):
                    continue
                pnames.append(pname)
            if pnames:
                layers_to_parameterize.append((sl, pnames))
            else:
                rest_of_layers.append(sl)
        return rest_of_layers, layers_to_parameterize

    def _coalesce_layers(
        sls: Iterable[tuple[Layer, list[str]]]
    ) -> dict[tuple[type[Layer], tuple[str, ...], tuple[tuple[int, ...], ...]], list[Layer]]:
        # Group layers having the same layer type, parameters to parameterize externally,
        # and shape for each of the parameters
        groups: dict[
            tuple[type[Layer], tuple[str, ...], tuple[tuple[int, ...], ...]], list[Layer]
        ] = defaultdict(list)
        for sl, pnames in sls:
            sl_settings = (type(sl), tuple(pnames), tuple(sl.params[p].shape for p in pnames))
            groups[sl_settings].append(sl)
        return groups

    # A map from symbolic layers in the input circuit to the symbolic layers in the output circuit
    layers_map: dict[Layer, Layer] = {}

    # A dictionary mapping each introduced parameter tensor name to the corresponding tensor shape
    function_specs: FunctionParameterSpecs = {}

    # Loop through each layer frontier obtained by the layerwise topological ordering
    group_id = 0
    for frontier in circuit.layerwise_topological_ordering():
        # Filter-out the layers in the frontier that do not have to be externally parameterize
        rest_of_frontier, filtered_frontier = _filter_layers(frontier)

        # Coalesce the layers based on (i) the layer class and (ii) the shape of their parameters
        coalesced_frontier = _coalesce_layers(filtered_frontier)

        # Iterate over the coalesced groups of layers in the filtered frontier
        for (sl_class, pnames, pshapes), group in coalesced_frontier.items():
            # Retrieve the layer class name
            sl_class_name = sl_class.__name__

            # Store the function specification
            function_parameter_names: dict[str, str] = {}
            for pname, pshape in zip(pnames, pshapes):
                fpname = f"g{group_id}.{sl_class_name}.{pname}"
                function_parameter_names[pname] = fpname
                # E.g., "g151.SumLayer.weight": (len(group), K_1, ..., K_n)
                function_specs[fpname] = (len(group), *pshape)

            # Iterate over the layers within each group,
            # and replace the parameter tensor to be externally parameterized
            for i, sl in enumerate(group):
                sl_updated_params = {
                    pname: Parameter.from_input(
                        FunctionParameter(
                            *pgraph.shape,
                            function=function,
                            name=function_parameter_names[pname],
                            index=i,
                        )
                    )
                    for pname, pgraph in sl.params.items()
                }
                layers_map[sl] = sl.copy(params=sl_updated_params)

            # Increment the layer group id
            group_id += 1

        # Shallow copy the layers that do not need to be re-parameterized
        for sl in rest_of_frontier:
            layers_map[sl] = sl.copy()

    # Construct the resulting circuit
    layers = list(layers_map.values())
    in_layers = {
        sl: [layers_map[prev_sli] for prev_sli in circuit.layer_inputs(prev_sl)]
        for prev_sl, sl in layers_map.items()
    }
    output_layers = [layers_map[prev_sli] for prev_sli in circuit.outputs]
    circuit = Circuit(circuit.num_channels, layers, in_layers=in_layers, outputs=output_layers)

    # Return both the resulting circuit and the function parameter specifications
    return circuit, function_specs
