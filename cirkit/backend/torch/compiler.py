import functools
import os
from collections import defaultdict
from itertools import chain
from typing import IO, Callable, Dict, List, Optional, Tuple, Union, cast

from torch import Tensor

from cirkit.backend.compiler import (
    AbstractCompiler,
    CompilerInitializerRegistry,
    CompilerLayerRegistry,
    CompilerParameterRegistry,
)
from cirkit.backend.torch.circuits import AbstractTorchCircuit, TorchCircuit, TorchConstantCircuit
from cirkit.backend.torch.graph.folding import build_folded_graph
from cirkit.backend.torch.graph.optimize import OptMatchStrategy, match_optimization_patterns
from cirkit.backend.torch.initializers import stacked_initializer_
from cirkit.backend.torch.layers import TorchLayer
from cirkit.backend.torch.optimizations.layers import (
    DEFAULT_CIRCUIT_OPT_RULES,
    DEFAULT_LAYER_OPT_RULES,
    CircuitOptMatch,
)
from cirkit.backend.torch.optimizations.parameters import DEFAULT_PARAMETER_OPT_RULES
from cirkit.backend.torch.parameters.leaves import TorchPointerParameter, TorchTensorParameter
from cirkit.backend.torch.parameters.parameter import (
    TorchParameter,
    TorchParameterNode,
    TorchParameterOp,
)
from cirkit.backend.torch.rules import (
    DEFAULT_INITIALIZER_COMPILATION_RULES,
    DEFAULT_LAYER_COMPILATION_RULES,
    DEFAULT_PARAMETER_COMPILATION_RULES,
)
from cirkit.backend.torch.semiring import Semiring, SemiringImpl
from cirkit.symbolic.circuit import Circuit, CircuitOperator, pipeline_topological_ordering
from cirkit.symbolic.initializers import Initializer
from cirkit.symbolic.layers import Layer
from cirkit.symbolic.parameters import Parameter, ParameterNode, TensorParameter


class TorchCompilerState:
    def __init__(self):
        # A map from symbolic parameter tensors to a tuple containing the compiled parameter tensor,
        # and the slice index, which is 0 if the compiled parameter tensor is unfolded.
        # If the compiled parameter tensor is folded, then the slice index can be non-zero.
        self._compiled_parameters: Dict[TensorParameter, Tuple[TorchTensorParameter, int]] = {}

        # We keep a reverse map from compiled and unfolded parameter tensors
        # to the corresponding symbolic parameter tensors.
        # This is useful to update the map from symbolic to compiled parameter tensors above
        # after we fold the tensor parameters within a circuit.
        # Since this is useful only for folding, it will be cleared after each circuit compilation.
        self._symbolic_parameters: Dict[TorchTensorParameter, TensorParameter] = {}

    def finish_compilation(self) -> None:
        # Clear the map from (unfolded) compiled parameter tensors to symbolic ones
        self._symbolic_parameters.clear()

    def retrieve_compiled_parameter(self, p: TensorParameter) -> Tuple[TorchTensorParameter, int]:
        # Retrieve the compiled parameter: we return the fold index as well.
        return self._compiled_parameters[p]

    def retrieve_symbolic_parameter(self, p: TorchTensorParameter) -> TensorParameter:
        # Retrieve the symbolic parameter tensor associated to the compiled one (which is unfolded)
        return self._symbolic_parameters[p]

    def register_compiled_parameter(
        self, sp: TensorParameter, cp: TorchTensorParameter, *, fold_idx: Optional[int] = None
    ) -> None:
        # Register a link from a symbolic parameter tensor to a compiled parameter tensor.
        if fold_idx is None:
            # We are registering an unfolded compiled parameter tensor
            # So, we can also register the reverse map (i.e., compiled to symbolic)
            self._compiled_parameters[sp] = (cp, 0)
            self._symbolic_parameters[cp] = sp

        # We are registering a folded compiled parameter tensor
        # So, we associate the symbolic parameter tensor to a particular slice of the
        # folded compiled parameter tensor, which is specified by the 'fold_idx'.
        self._compiled_parameters[sp] = (cp, fold_idx)


class TorchCompiler(AbstractCompiler):
    def __init__(self, semiring: str = "sum-product", fold: bool = False, optimize: bool = False):
        super().__init__(
            CompilerLayerRegistry(DEFAULT_LAYER_COMPILATION_RULES),
            CompilerParameterRegistry(DEFAULT_PARAMETER_COMPILATION_RULES),
            CompilerInitializerRegistry(DEFAULT_INITIALIZER_COMPILATION_RULES),
            fold=fold,
            optimize=optimize,
        )

        # The semiring being used at compile time
        self._semiring: Semiring = SemiringImpl.from_name(semiring)

        # The state of the compiler
        self._state = TorchCompilerState()

        # The registry of optimization rules
        # TODO: make these rules extendable from outside
        #       by wrapping each of them in a compiler registry
        self._optimization_rules = {
            "circuit": DEFAULT_CIRCUIT_OPT_RULES,
            "parameter": DEFAULT_PARAMETER_OPT_RULES,
            "layer": DEFAULT_LAYER_OPT_RULES,
        }

    def compile_pipeline(self, sc: Circuit) -> AbstractTorchCircuit:
        # Compile the circuits following the topological ordering of the pipeline.
        for sci in pipeline_topological_ordering([sc]):
            # Check if the circuit in the pipeline has already been compiled
            if self.is_compiled(sci):
                continue

            # Compile the circuit
            self._compile_circuit(sci)

        # Return the compiled circuit (i.e., the output of the circuit pipeline)
        return self.get_compiled_circuit(sc)

    @property
    def semiring(self) -> Semiring:
        return self._semiring

    @property
    def is_fold_enabled(self) -> bool:
        return self._flags["fold"]

    @property
    def is_optimize_enabled(self) -> bool:
        return self._flags["optimize"]

    @property
    def state(self) -> TorchCompilerState:
        return self._state

    def compile_parameter(self, parameter: Parameter) -> TorchParameter:
        # A map from symbolic to compiled parameters
        compiled_nodes_map: Dict[ParameterNode, TorchParameterNode] = {}

        # The parameter nodes, and their inputs and outputs
        nodes: List[TorchParameterNode] = []
        in_nodes: Dict[TorchParameterNode, List[TorchParameterNode]] = {}
        out_nodes: Dict[TorchParameterNode, List[TorchParameterNode]] = defaultdict(list)

        # Compile the parameter by following the topological ordering
        for p in parameter.topological_ordering():
            # Compile the parameter node and make the connections
            compiled_p = self._compile_parameter_node(p)
            in_compiled_nodes = [compiled_nodes_map[pi] for pi in parameter.node_inputs(p)]
            in_nodes[compiled_p] = in_compiled_nodes
            for pi in in_compiled_nodes:
                out_nodes[pi].append(compiled_p)
            compiled_nodes_map[p] = compiled_p
            nodes.append(compiled_p)

        # Build the parameter's computational graph
        return TorchParameter(nodes, in_nodes, out_nodes, topologically_ordered=True)

    def compiler_initializer(self, initializer: Initializer) -> Callable[[Tensor], Tensor]:
        # Retrieve the rule for the given initializer and compile it
        signature = type(initializer)
        rule = self.retrieve_initializer_rule(signature)
        return cast(Callable[[Tensor], Tensor], rule(self, initializer))

    def _compile_layer(self, layer: Layer) -> TorchLayer:
        signature = type(layer)
        rule = self.retrieve_layer_rule(signature)
        return cast(TorchLayer, rule(self, layer))

    def _compile_parameter_node(self, node: ParameterNode) -> TorchParameterNode:
        signature = type(node)
        rule = self.retrieve_parameter_rule(signature)
        return cast(TorchParameterNode, rule(self, node))

    def _compile_circuit(self, sc: Circuit) -> AbstractTorchCircuit:
        # A map from symbolic to compiled layers
        compiled_layers_map: Dict[Layer, TorchLayer] = {}

        # The inputs and outputs for each layer
        in_layers: Dict[TorchLayer, List[TorchLayer]] = {}
        out_layers: Dict[TorchLayer, List[TorchLayer]] = defaultdict(list)

        # Compile layers by following the topological ordering
        for sl in sc.topological_ordering():
            # Compile the layer, for any layer types
            layer = self._compile_layer(sl)

            # Build the connectivity between compiled layers
            ins = [compiled_layers_map[sli] for sli in sc.layer_inputs(sl)]
            in_layers[layer] = ins
            for li in ins:
                out_layers[li].append(layer)
            compiled_layers_map[sl] = layer

        # If the symbolic circuit being compiled has been obtained by integrating
        # another circuit over all the variables it is defined on,
        # then return a 'constant circuit' whose interface does not require inputs
        if (
            sc.operation is not None
            and sc.operation.operator == CircuitOperator.INTEGRATION
            and sc.operation.metadata["scope"] == sc.scope
        ):
            cc_cls = TorchConstantCircuit
        else:
            cc_cls = TorchCircuit

        # Construct the tensorized circuit
        layers = [compiled_layers_map[sl] for sl in compiled_layers_map.keys()]
        cc = cc_cls(
            sc.scope,
            sc.num_channels,
            layers=layers,
            in_layers=in_layers,
            out_layers=out_layers,
            topologically_ordered=True,
        )

        # Post-process the compiled circuit, i.e.,
        # optionally apply optimizations to it and then fold it
        cc = self._post_process_circuit(cc)

        # Initialize some stuff
        cc.reset_parameters()
        cc.initialize_address_book()

        # Register the compiled circuit
        self.register_compiled_circuit(sc, cc)

        # Signal the end of the circuit compilation to the state
        self._state.finish_compilation()
        return cc

    def _post_process_circuit(self, cc: AbstractTorchCircuit) -> AbstractTorchCircuit:
        if self.is_optimize_enabled:
            # Optimize the circuit computational graph
            opt_cc = _optimize_circuit(self, cc)
            del cc
            cc = opt_cc
        if self.is_fold_enabled:
            # Optimize the circuit by folding it
            opt_cc = _fold_circuit(self, cc)
            del cc
            cc = opt_cc
        return cc

    def save(
        self,
        sym_filepath: Union[IO, os.PathLike, str],
        compiled_filepath: Union[IO, os.PathLike, str],
    ):
        ...

    @staticmethod
    def load(
        sym_filepath: Union[IO, os.PathLike, str], tens_filepath: Union[IO, os.PathLike, str]
    ) -> "TorchCompiler":
        ...


def _fold_circuit(compiler: TorchCompiler, cc: AbstractTorchCircuit) -> AbstractTorchCircuit:
    # Fold the layers in the given circuit, by following the layer-wise topological ordering
    layers, in_layers, out_layers, fold_idx_info = build_folded_graph(
        cc.layerwise_topological_ordering(),
        outputs=cc.outputs,
        incomings_fn=cc.layer_inputs,
        fold_group_fn=functools.partial(_fold_layers_group, compiler=compiler),
        in_address_fn=lambda l: l.scope,
    )

    # Instantiate a folded circuit
    return type(cc)(
        cc.scope,
        cc.num_channels,
        layers,
        in_layers,
        out_layers,
        topologically_ordered=True,
        fold_idx_info=fold_idx_info,
    )


def _fold_layers_group(layers: List[TorchLayer], *, compiler: TorchCompiler) -> TorchLayer:
    # Retrieve the class of the folded layer, as well as the configuration attributes
    fold_layer_cls = type(layers[0])
    fold_layer_conf = layers[0].config
    num_folds = len(layers)
    fold_layer_conf.update(num_folds=num_folds)

    # Retrieve the parameters of each layer
    layer_params: Dict[str, List[TorchParameter]] = defaultdict(list)
    for l in layers:
        for n, p in l.params.items():
            layer_params[n].append(p)

    # Fold the parameters, if the layers have any
    fold_layer_parameters: Dict[str, TorchParameter] = {
        n: _fold_parameters(compiler, ps) for n, ps in layer_params.items()
    }

    # Instantiate a new folded layer, using the folded layer configuration and the folded parameters
    return fold_layer_cls(**fold_layer_conf, **fold_layer_parameters, semiring=compiler.semiring)


def _fold_parameters(compiler: TorchCompiler, parameters: List[TorchParameter]) -> TorchParameter:
    # Retrieve:
    # (i)  the parameter nodes and the input to each node;
    # (ii) the layer-wise (aka bottom-up) topological orderings of parameter nodes
    in_nodes: Dict[TorchParameterNode, List[TorchParameterNode]] = {}
    for pi in parameters:
        in_nodes.update(pi.nodes_inputs)
    ordering: List[List[TorchParameterNode]] = []
    for pi in parameters:
        for i, frontier in enumerate(pi.layerwise_topological_ordering()):
            if i < len(ordering):
                ordering[i].extend(frontier)
                continue
            ordering.append(frontier)

    # Fold the nodes in the merged parameter computational graphs,
    # by following the layer-wise topological ordering
    nodes, in_nodes, out_nodes, fold_idx_info = build_folded_graph(
        ordering,
        outputs=chain.from_iterable(map(lambda pi: pi.outputs, parameters)),
        incomings_fn=in_nodes.get,
        fold_group_fn=functools.partial(_fold_parameter_nodes_group, compiler=compiler),
    )

    # Construct the folded parameter's computational graph
    return TorchParameter(
        nodes, in_nodes, out_nodes, topologically_ordered=True, fold_idx_info=fold_idx_info
    )


def _fold_parameter_nodes_group(
    group: List[TorchParameterNode], *, compiler: TorchCompiler
) -> TorchParameterNode:
    fold_node_cls = type(group[0])
    # Catch the case we are folding tensor parameters
    # That is, we set the number of folds, copy the number of parameters and relevant flags,
    # and stack the initialization functions together.
    if issubclass(fold_node_cls, TorchTensorParameter):
        assert all(isinstance(p, TorchTensorParameter) for p in group)
        folded_node = TorchTensorParameter(
            *group[0].shape,
            num_folds=len(group),
            requires_grad=group[0].requires_grad,
            initializer_=functools.partial(
                stacked_initializer_, initializers=list(map(lambda p: p.initializer_, group))
            ),
        )
        # If we are folding parameter tensors, then update the registry as to maintain the correct
        # mapping between symbolic parameter leaves (which are unfolded) and slices within the folded
        # compiled parameter leaves.
        for i, p in enumerate(group):
            sp = compiler.state.retrieve_symbolic_parameter(p)
            compiler.state.register_compiled_parameter(sp, folded_node, fold_idx=i)
        return folded_node
    # Catch the case we are folding parameters obtained via slicing
    # This case regularly fires when doing operations over circuits
    # that are compiled into folded tensorized circuits
    if issubclass(fold_node_cls, TorchPointerParameter):
        assert all(isinstance(p, TorchPointerParameter) for p in group)
        if len(group) == 1:
            # Catch the case we are not able to fold multiple tensor slicing operations
            # In such a case, just have the slice as folded parameter (i.e., number of folds = 1)
            return group[0]
        # Catch the case we are able to fold multiple tensor slicing operations
        in_folded_node = group[0].deref()
        in_fold_idx: List[int] = list(
            chain.from_iterable(
                list(range(p.num_folds)) if p.fold_idx is None else p.fold_idx for p in group
            )
        )
        return TorchPointerParameter(in_folded_node, fold_idx=in_fold_idx)
    # We are folding an operator: just set the number of folds and copy the configuration parameters
    assert all(isinstance(p, TorchParameterOp) for p in group)
    return fold_node_cls(*group[0].in_shapes, num_folds=len(group), **group[0].config)


def _optimize_circuit(
    compiler: TorchCompiler, cc: AbstractTorchCircuit, *, max_opt_steps: int = 1
) -> AbstractTorchCircuit:
    assert max_opt_steps > 0

    # Each optimization step consists of three kinds of optimizations (see below).
    # We continue optimizing until no further optimization can be performed
    # or if we reach a maximum number of optimization steps being performed
    optimizing = True
    opt_step = 0
    while optimizing and opt_step < max_opt_steps:
        # First optimization step: fuse the parameters node of the parameter graphs of each layer
        opt_cc, opt_fuse_parameter_nodes = _optimize_fuse_parameter_nodes(compiler, cc)
        del cc
        cc = opt_cc

        # Second optimization step: shatter layers in multiple more efficient ones
        opt_cc, opt_shatter_layers = _optimize_shatter_layers(compiler, cc)
        del cc
        cc = opt_cc

        # Third optimization step: fuse multiple layers into a single more efficient one
        opt_cc, opt_fuse_layers = _optimize_fuse_layers(compiler, cc)
        del cc
        cc = opt_cc

        # Update the optimization step and whether we should continue optimizing
        optimizing = opt_fuse_parameter_nodes or opt_shatter_layers or opt_fuse_layers
        opt_step += 1

    return cc


def _optimize_fuse_layers(
    compiler: TorchCompiler, cc: AbstractTorchCircuit
) -> Tuple[AbstractTorchCircuit, bool]:
    # Match optimization patterns
    # matches: list of all matched and grounded optimization rules
    # module_matches: a map from modules to the matches they belong to, if any
    circuit_optimization_rules = compiler._optimization_rules["circuit"]
    matches, layer_matches = match_optimization_patterns(
        cc, circuit_optimization_rules.keys(), strategy=OptMatchStrategy.LARGEST_MATCH
    )

    # Run the matched optimization rules and collect the optimized layers
    optimized_layers: Dict[CircuitOptMatch, TorchLayer] = {}
    for match in matches:
        opt_func = circuit_optimization_rules[match.pattern]
        opt_layer = opt_func(compiler, match)
        optimized_layers[match] = opt_layer

    # The list of optimized layer and the inputs/outputs of each optimized layer
    layers: List[TorchLayer] = []
    in_layers: Dict[TorchLayer, List[TorchLayer]] = {}
    out_layers: Dict[TorchLayer, List[TorchLayer]] = defaultdict(list)

    # A map from matches to their entry point unoptimized layers
    match_entries: Dict[CircuitOptMatch, TorchLayer] = {}

    # Build the optimize circuit by following the topological ordering
    for layer in cc.topological_ordering():
        match = layer_matches.get(layer, None)

        # Check if the layer does not belong to any matched pattern
        # If so, then just add it to the optimize layer as is
        if match is None:
            layers.append(layer)
            layer_ins = [
                optimized_layers[layer_matches[li]] if li in layer_matches else li
                for li in cc.layer_inputs(layer)
            ]
            in_layers[layer] = layer_ins
            for li in layer_ins:
                out_layers[li].append(layer)
            continue

        # Check if the layer is the root within the matched pattern
        # If so, then add the corresponding sub-computational-graph optimization to the
        # optimized circuit, and build connections
        if layer == match.entries[0]:
            opt_layer = optimized_layers[match]
            match_entry = match_entries[match]
            layers.append(opt_layer)
            opt_layer_ins = [
                optimized_layers[layer_matches[li]] if li in layer_matches else li
                for li in cc.layer_inputs(match_entry)
            ]
            in_layers[opt_layer] = opt_layer_ins
            for li in opt_layer_ins:
                out_layers[li].append(opt_layer)
            continue

        # If the layer belongs to a matched pattern (there can only be a single one by construction),
        # but it is not the root in that pattern,
        # then register it as the entry point of the matched sub-computational-graph, if not other entry
        # point has been registered before
        # This is necessary as to recover the connectivity between optimized layers
        # whose roots are far away in the unoptimized computational graph
        # TODO: generalize this as to cover patterns with multiply entry points
        if match not in match_entries:
            match_entries[match] = layer

    opt_cc = type(cc)(
        cc.scope, cc.num_channels, layers, in_layers, out_layers, topologically_ordered=True
    )
    return opt_cc, True


def _optimize_fuse_parameter_nodes(
    compiler: TorchCompiler, cc: AbstractTorchCircuit
) -> Tuple[AbstractTorchCircuit, bool]:
    return cc, False


def _optimize_shatter_layers(
    compiler: TorchCompiler, cc: AbstractTorchCircuit
) -> Tuple[AbstractTorchCircuit, bool]:
    return cc, False
