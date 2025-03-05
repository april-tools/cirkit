import functools
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import chain
from typing import cast

import torch
from torch import Tensor, nn

from cirkit.backend.compiler import (
    AbstractCompiler,
    CompilerInitializerRegistry,
    CompilerLayerRegistry,
    CompilerParameterRegistry,
)
from cirkit.backend.registry import CompilerRegistry
from cirkit.backend.torch.circuits import (
    AbstractTorchCircuit,
    TorchCircuit,
    TorchConditionalCircuit,
    TorchConstantCircuit,
)
from cirkit.backend.torch.graph.folding import build_folded_graph
from cirkit.backend.torch.graph.optimize import (
    GraphOptPattern,
    match_optimization_patterns,
    optimize_graph,
)
from cirkit.backend.torch.initializers import foldwise_initializer_
from cirkit.backend.torch.layers import TorchInputLayer, TorchLayer
from cirkit.backend.torch.layers.input import TorchConstantLayer
from cirkit.backend.torch.optimization.layers import (
    DEFAULT_LAYER_FUSE_OPT_RULES,
    DEFAULT_LAYER_SHATTER_OPT_RULES,
)
from cirkit.backend.torch.optimization.parameters import DEFAULT_PARAMETER_OPT_RULES
from cirkit.backend.torch.optimization.registry import (
    LayerOptApplyFunc,
    LayerOptMatch,
    LayerOptPattern,
    LayerOptRegistry,
    ParameterOptApplyFunc,
    ParameterOptMatch,
    ParameterOptPattern,
    ParameterOptRegistry,
)
from cirkit.backend.torch.parameters.nodes import (
    TorchGateFunctionParameter,
    TorchParameterNode,
    TorchParameterOp,
    TorchPointerParameter,
    TorchTensorParameter,
)
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.rules import (
    DEFAULT_INITIALIZER_COMPILATION_RULES,
    DEFAULT_LAYER_COMPILATION_RULES,
    DEFAULT_PARAMETER_COMPILATION_RULES,
)
from cirkit.backend.torch.semiring import Semiring, SemiringImpl
from cirkit.backend.torch.utils import CachedGateFunctionEval
from cirkit.symbolic.circuit import Circuit, ConditionalCircuit, pipeline_topological_ordering
from cirkit.symbolic.initializers import Initializer
from cirkit.symbolic.layers import Layer
from cirkit.symbolic.parameters import Parameter, ParameterNode, TensorParameter


class TorchCompilerState:
    def __init__(self):
        # A map from symbolic parameter tensors to a tuple containing the compiled parameter tensor,
        # and the slice index, which is 0 if the compiled parameter tensor is unfolded.
        # If the compiled parameter tensor is folded, then the slice index can be non-zero.
        self._compiled_parameters: dict[TensorParameter, tuple[TorchTensorParameter, int]] = {}

        # We keep a reverse map from compiled and unfolded parameter tensors
        # to the corresponding symbolic parameter tensors.
        # This is useful to update the map from symbolic to compiled parameter tensors above
        # after we fold the tensor parameters within a circuit.
        # Since this is useful only for folding, it will be cleared after each circuit compilation.
        self._symbolic_parameters: dict[TorchTensorParameter, TensorParameter] = {}

        # A map from external gate functions identifiers to a tuple containing the object used to evaluate them
        # and the shape expected from it
        self._gate_functions_evals: dict[str, tuple[tuple[int, ...], CachedGateFunctionEval]] = {}

    @property
    def gate_functions(self) -> Mapping[str, tuple[tuple[int, ...], CachedGateFunctionEval]]:
        return self._gate_functions_evals

    def finish_compilation(self):
        # Clear the map from (unfolded) compiled parameter tensors to symbolic ones
        self._symbolic_parameters = {}

        # Clear the map of gate functions
        self._gate_functions_evals = {}

    def has_compiled_parameter(self, p: TensorParameter) -> bool:
        # Retrieve whether a tensor parameter has already been compiled
        return p in self._compiled_parameters

    def has_gate_function(self, gate_function_id: str) -> bool:
        # Retrieve whether a gate function has already been compiled
        return gate_function_id in self._gate_functions_evals

    def retrieve_compiled_parameter(self, p: TensorParameter) -> tuple[TorchTensorParameter, int]:
        # Retrieve the compiled parameter: we return the fold index as well.
        return self._compiled_parameters[p]

    def retrieve_symbolic_parameter(self, p: TorchTensorParameter) -> TensorParameter:
        # Retrieve the symbolic parameter tensor associated to the compiled one (which is unfolded)
        return self._symbolic_parameters[p]

    def retrieve_gate_function(
        self, gate_function_id: str
    ) -> tuple[tuple[int, ...], CachedGateFunctionEval]:
        # Retrieve the external gate function evaluator
        return self._gate_functions_evals[gate_function_id]

    def register_compiled_parameter(
        self, sp: TensorParameter, cp: TorchTensorParameter, *, fold_idx: int | None = None
    ):
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

    def register_gate_function(self, name: str, gate_function_eval: CachedGateFunctionEval):
        # Register the gate function evaluator to the running state of the compiler
        self._gate_functions_evals[name] = gate_function_eval


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
        self._optimization_registry = {
            "parameter": ParameterOptRegistry(DEFAULT_PARAMETER_OPT_RULES),
            "layer_fuse": LayerOptRegistry(DEFAULT_LAYER_FUSE_OPT_RULES),
            "layer_shatter": LayerOptRegistry(DEFAULT_LAYER_SHATTER_OPT_RULES),
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

    def compile_layer(self, layer: Layer) -> TorchLayer:
        signature = type(layer)
        rule = self.retrieve_layer_rule(signature)
        return cast(TorchLayer, rule(self, layer))

    def compile_parameter(self, parameter: Parameter) -> TorchParameter:
        # A map from symbolic to compiled parameters
        compiled_nodes_map: dict[ParameterNode, TorchParameterNode] = {}

        # The parameter nodes, and their inputs
        nodes: list[TorchParameterNode] = []
        in_nodes: dict[TorchParameterNode, list[TorchParameterNode]] = {}

        # Compile the parameter by following the topological ordering
        for p in parameter.topological_ordering():
            # Compile the parameter node and make the connections
            compiled_p = self._compile_parameter_node(p)
            in_compiled_nodes = [compiled_nodes_map[pi] for pi in parameter.node_inputs(p)]
            in_nodes[compiled_p] = in_compiled_nodes
            compiled_nodes_map[p] = compiled_p
            nodes.append(compiled_p)

        # Build the parameter's computational graph
        outputs = [compiled_nodes_map[parameter.output]]
        return TorchParameter(nodes, in_nodes, outputs)

    def compile_initializer(self, initializer: Initializer) -> Callable[[Tensor], Tensor]:
        # Retrieve the rule for the given initializer and compile it
        signature = type(initializer)
        rule = self.retrieve_initializer_rule(signature)
        return cast(Callable[[Tensor], Tensor], rule(self, initializer))

    def retrieve_optimization_registry(self, kind: str) -> CompilerRegistry:
        return cast(CompilerRegistry, self._optimization_registry[kind])

    def retrieve_optimization_rule(self, kind: str, pattern: GraphOptPattern) -> Callable:
        registry = self.retrieve_optimization_registry(kind)
        return registry.retrieve_rule(pattern)

    def _compile_parameter_node(self, node: ParameterNode) -> TorchParameterNode:
        signature = type(node)
        rule = self.retrieve_parameter_rule(signature)
        return cast(TorchParameterNode, rule(self, node))

    def _compile_circuit(self, sc: Circuit) -> AbstractTorchCircuit:
        # A map from symbolic to compiled layers
        compiled_layers_map: dict[Layer, TorchLayer] = {}

        # The inputs of each layer
        in_layers: dict[TorchLayer, list[TorchLayer]] = {}

        # Compile layers by following the topological ordering
        for sl in sc.topological_ordering():
            # Compile the layer, for any layer types
            layer = self.compile_layer(sl)

            # Build the connectivity between compiled layers
            ins = [compiled_layers_map[sli] for sli in sc.layer_inputs(sl)]
            in_layers[layer] = ins
            compiled_layers_map[sl] = layer

        # Construct the sequence of output layers
        outputs = [compiled_layers_map[sl] for sl in sc.outputs]

        # Collect the layers for the tensorized circuit
        layers = list(compiled_layers_map.values())

        # Prepare the arguments for the torch circuit
        cc_kwargs = {
            "scope": sc.scope,
            "layers": layers,
            "in_layers": in_layers,
            "outputs": outputs,
            "properties": sc.properties,
        }

        if isinstance(sc, ConditionalCircuit):
            # Compile a conditional circuit if the symbolic one is conditional
            cc_cls = TorchConditionalCircuit
            cc_kwargs["gate_function_evals"] = self._state.gate_functions
            cc_kwargs["gate_function_specs"] = sc.gate_function_specs
        elif not sc.scope:
            # If the symbolic circuit being compiled has empty scope,
            # then return a 'constant circuit' whose interface does not require inputs
            cc_cls = TorchConstantCircuit
        else:
            cc_cls = TorchCircuit

        # Instantiate the circuit
        cc = cc_cls(**cc_kwargs)

        # Post-process the compiled circuit, i.e.,
        # optionally apply optimizations to it and then fold it
        cc = self._post_process_circuit(cc)

        # Allocate & initialize the parameters
        cc.reset_parameters()

        # Register the compiled circuit
        self.register_compiled_circuit(sc, cc)

        # Signal the end of the circuit compilation to the state
        self._state.finish_compilation()
        return cc

    def _post_process_circuit(self, cc: AbstractTorchCircuit) -> AbstractTorchCircuit:
        if self.is_optimize_enabled:
            # Optimize the circuit computational graph
            opt_cc = _optimize_circuit(self, cc, max_opt_steps=5)
            del cc
            cc = opt_cc
        if self.is_fold_enabled:
            # Optimize the circuit by folding it
            opt_cc = _fold_circuit(self, cc)
            del cc
            cc = opt_cc
        return cc


def _fold_circuit(compiler: TorchCompiler, cc: AbstractTorchCircuit) -> AbstractTorchCircuit:
    # Fold the layers in the given circuit, by following the layer-wise topological ordering
    layers, in_layers, outputs, fold_idx_info = build_folded_graph(
        cc.layerwise_topological_ordering(),
        outputs=cc.outputs,
        incomings_fn=cc.layer_inputs,
        fold_group_fn=functools.partial(_fold_layers_group, compiler=compiler),
    )

    # Instantiate a folded circuit
    cc_kwargs = {
        "scope": cc.scope,
        "layers": layers,
        "in_layers": in_layers,
        "outputs": outputs,
        "properties": cc.properties,
        "fold_idx_info": fold_idx_info,
    }

    if isinstance(cc, TorchConditionalCircuit):
        cc_kwargs["gate_function_evals"] = cc.gate_function_evals
        cc_kwargs["gate_function_specs"] = cc.gate_function_specs

    cc = type(cc)(**cc_kwargs)
    return cc


def _fold_layers_group(layers: list[TorchLayer], *, compiler: TorchCompiler) -> TorchLayer:
    # Retrieve the class of the folded layer, as well as the configuration attributes
    fold_layer_cls = type(layers[0])
    fold_layer_conf = layers[0].config

    # If we are folding input layers, then concatenate the variables scope index tensors
    kwargs = {}
    if issubclass(fold_layer_cls, TorchInputLayer):
        if not issubclass(fold_layer_cls, TorchConstantLayer):
            kwargs["scope_idx"] = torch.cat([l.scope_idx for l in layers])
    else:
        # We are folding sum or product layers, so simply set the number of folds
        kwargs["num_folds"] = sum(l.num_folds for l in layers)

    # Retrieve the parameters of each layer, and
    # retrieve the sub-module layers of each layer
    layer_params: dict[str, list[TorchParameter]] = defaultdict(list)
    layer_submodules: dict[str, list[TorchLayer]] = defaultdict(list)
    for l in layers:
        for n, p in l.params.items():
            layer_params[n].append(p)
        for n, sub_l in l.sub_modules.items():
            layer_submodules[n].append(sub_l)

    # Fold the parameters, if the layers have any
    fold_layer_parameters: dict[str, TorchParameter] = {
        n: _fold_parameters(compiler, ps) for n, ps in layer_params.items()
    }

    # Fold all sub-module layers, if the layers have any
    fold_layer_submodules: dict[str, TorchLayer] = {
        n: _fold_layers_group(ls, compiler=compiler) for n, ls in layer_submodules.items()
    }

    # Instantiate a new folded layer, using the folded layer configuration and the folded parameters
    return fold_layer_cls(
        **fold_layer_conf,
        **fold_layer_submodules,
        **fold_layer_parameters,
        semiring=compiler.semiring,
        **kwargs,
    )


def _fold_parameters(compiler: TorchCompiler, parameters: list[TorchParameter]) -> TorchParameter:
    # Retrieve:
    # (i)  the parameter nodes and the input to each node;
    # (ii) the layer-wise (aka bottom-up) topological orderings of parameter nodes
    in_nodes: dict[TorchParameterNode, Sequence[TorchParameterNode]] = {}
    for pi in parameters:
        in_nodes.update(pi.nodes_inputs)
    ordering: list[list[TorchParameterNode]] = []
    for pi in parameters:
        for i, frontier in enumerate(pi.layerwise_topological_ordering()):
            if i < len(ordering):
                ordering[i].extend(frontier)
                continue
            ordering.append(frontier)

    # Fold the nodes in the merged parameter computational graphs,
    # by following the layer-wise topological ordering
    nodes, in_nodes, outputs, fold_idx_info = build_folded_graph(
        ordering,
        outputs=chain.from_iterable(map(lambda pi: pi.outputs, parameters)),
        incomings_fn=in_nodes.get,
        fold_group_fn=functools.partial(_fold_parameter_nodes_group, compiler=compiler),
    )

    # Construct the folded parameter's computational graph
    return TorchParameter(nodes, in_nodes, outputs, fold_idx_info=fold_idx_info)


def _fold_parameter_nodes_group(
    group: list[TorchParameterNode], *, compiler: TorchCompiler
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
                foldwise_initializer_, initializers=list(map(lambda p: p.initializer, group))
            ),
            dtype=group[0].dtype,
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
        in_fold_idx: list[int] = list(
            chain.from_iterable(
                list(range(p.num_folds)) if p.fold_idx is None else p.fold_idx for p in group
            )
        )
        return TorchPointerParameter(in_folded_node, fold_idx=in_fold_idx)
    # Catch the case we are folding parameters obtained from an external function
    if issubclass(fold_node_cls, TorchGateFunctionParameter):
        assert all(isinstance(p, TorchGateFunctionParameter) for p in group)
        if len(group) == 1:
            # Catch the case we are folding a single torch function parameter
            # In such a case, we just return it as it is
            return group[0]
        # Catch the case we are folding multiple torch function parameters
        fold_idx: list[int] = list(chain.from_iterable(p.fold_idx for p in group))
        return TorchGateFunctionParameter(
            *group[0].shape,
            gate_function_eval=group[0].gate_function_eval,
            name=group[0].name,
            fold_idx=fold_idx,
        )
    # We are folding an operator: just set the number of folds and copy the configuration parameters
    assert all(isinstance(p, TorchParameterOp) for p in group)
    return fold_node_cls(**group[0].config, num_folds=len(group))


def _optimize_circuit(
    compiler: TorchCompiler, cc: AbstractTorchCircuit, *, max_opt_steps: int = 5
) -> AbstractTorchCircuit:
    assert max_opt_steps > 0

    # Each optimization step consists of three kinds of optimizations (see below).
    # We continue optimizing until no further optimization can be performed
    # or if we reach a maximum number of optimization steps being performed
    optimizing = True
    opt_step = 0
    while optimizing and opt_step < max_opt_steps:
        # First optimization step: optimize the parameters node of the parameter graphs of each layer
        opt_cc, opt_fuse_parameter_nodes = _optimize_parameter_nodes(compiler, cc)
        del cc
        cc = opt_cc

        # Second optimization step: shatter layers in multiple more efficient ones
        opt_cc, opt_shatter_layers = _optimize_layers(compiler, cc, shatter=True)
        del cc
        cc = opt_cc

        # Third optimization step: fuse multiple layers into a single more efficient one
        opt_cc, opt_fuse_layers = _optimize_layers(compiler, cc, shatter=False)
        del cc
        cc = opt_cc

        # Update the optimization step and whether we should continue optimizing
        optimizing = opt_fuse_parameter_nodes or opt_shatter_layers or opt_fuse_layers
        opt_step += 1

    return cc


def _optimize_parameter_nodes(
    compiler: TorchCompiler, cc: AbstractTorchCircuit
) -> tuple[AbstractTorchCircuit, bool]:
    def match_optimizer(match: ParameterOptMatch) -> tuple[TorchParameterNode, ...]:
        rule = compiler.retrieve_optimization_rule("parameter", match.pattern)
        func = cast(ParameterOptApplyFunc, rule)
        return func(compiler, match)

    # Loop through all the layers
    has_been_optimized = False
    patterns = compiler.retrieve_optimization_registry("parameter").signatures
    for layer in cc.layers:
        # Retrieve the parameter computational graphs of the layer
        for pname, pgraph in layer.params.items():
            # Optimize the parameter computational graph
            optimize_result = optimize_graph(
                pgraph.topological_ordering(),
                pgraph.outputs,
                patterns,
                incomings_fn=pgraph.node_inputs,
                outcomings_fn=pgraph.node_outputs,
                pattern_matcher_fn=_match_parameter_nodes_pattern,
                match_optimizer_fn=match_optimizer,
            )

            # Check if no optimization is possible
            if optimize_result is None:
                continue
            nodes, in_nodes, outputs = optimize_result

            # Build the optimized computational graph
            pgraph = type(pgraph)(nodes, in_nodes, outputs)

            # Update the parameter computational graph assigned to the layer
            assert hasattr(layer, pname)
            setattr(layer, pname, pgraph)
            has_been_optimized = True

    # Check whether no parameter optimization has been possible
    if has_been_optimized:
        return cc, True
    return cc, False


def _optimize_layers(
    compiler: TorchCompiler, cc: AbstractTorchCircuit, *, shatter: bool = False
) -> tuple[AbstractTorchCircuit, bool]:
    def match_optimizer_shatter(match: LayerOptMatch) -> tuple[TorchLayer, ...]:
        rule = compiler.retrieve_optimization_rule("layer_shatter", match.pattern)
        func = cast(LayerOptApplyFunc, rule)
        return func(compiler, match)

    def match_optimizer_fuse(match: LayerOptMatch) -> tuple[TorchLayer, ...]:
        rule = compiler.retrieve_optimization_rule("layer_fuse", match.pattern)
        func = cast(LayerOptApplyFunc, rule)
        return func(compiler, match)

    registry = compiler.retrieve_optimization_registry("layer_shatter" if shatter else "layer_fuse")
    match_optimizer = match_optimizer_shatter if shatter else match_optimizer_fuse
    optimize_result = optimize_graph(
        cc.topological_ordering(),
        cc.outputs,
        registry.signatures,
        incomings_fn=cc.layer_inputs,
        outcomings_fn=cc.layer_outputs,
        pattern_matcher_fn=_match_layer_pattern,
        match_optimizer_fn=match_optimizer,
    )
    if optimize_result is None:
        return cc, False
    layers, in_layers, outputs = optimize_result

    cc_kwargs = {
        "scope": cc.scope,
        "layers": layers,
        "in_layers": in_layers,
        "outputs": outputs,
        "properties": cc.properties,
    }

    if isinstance(cc, TorchConditionalCircuit):
        cc_kwargs["gate_function_evals"] = cc.gate_function_evals
        cc_kwargs["gate_function_specs"] = cc.gate_function_specs

    cc = type(cc)(**cc_kwargs)
    return cc, True


def _match_parameter_nodes_pattern(
    node: TorchParameterNode,
    pattern: ParameterOptPattern,
    *,
    incomings_fn: Callable[[TorchParameterNode], Sequence[TorchParameterNode]],
    outcomings_fn: Callable[[TorchParameterNode], Sequence[TorchParameterNode]],
) -> ParameterOptMatch | None:
    pattern_entries = pattern.entries()
    num_entries = len(pattern_entries)
    matched_nodes = []

    # Start matching the pattern from the root
    # TODO: generalize to match DAGs or binary trees
    for nid in range(num_entries):
        if not isinstance(node, pattern_entries[nid]):
            return None
        in_nodes = incomings_fn(node)
        if len(in_nodes) > 1 and nid != num_entries - 1:
            return None
        out_nodes = outcomings_fn(node)
        if len(out_nodes) > 1 and nid != 0:
            return None
        matched_nodes.append(node)
        if nid != num_entries - 1:
            (node,) = in_nodes

    return ParameterOptMatch(pattern, matched_nodes)


def _match_layer_pattern(
    layer: TorchLayer,
    pattern: LayerOptPattern,
    *,
    incomings_fn: Callable[[TorchLayer], Sequence[TorchLayer]],
    outcomings_fn: Callable[[TorchLayer], Sequence[TorchLayer]],
) -> LayerOptMatch | None:
    ppatterns = pattern.ppatterns()
    cpatterns = pattern.cpatterns()
    pattern_entries = pattern.entries()
    num_entries = len(pattern_entries)
    matched_layers = []
    matched_parameters = []

    # Start matching the pattern from the root
    # TODO: generalize to match DAGs or trees
    for lid in range(num_entries):
        # First, attempt to match the layer
        if not isinstance(layer, pattern_entries[lid]):
            return None
        in_nodes = incomings_fn(layer)
        if len(in_nodes) > 1 and lid != num_entries - 1:
            return None
        out_nodes = outcomings_fn(layer)
        if len(out_nodes) > 1 and lid != 0:
            return None

        # Second, attempt to match the configuration patterns for the layer
        for cname, cvalue in cpatterns[lid].items():
            if layer.config[cname] != cvalue:
                return None

        # Third, attempt to match the patterns specified for its parameters
        lpmatches = {}
        for pname, ppattern in ppatterns[lid].items():
            pgraph = layer.params[pname]
            matches, _ = match_optimization_patterns(
                pgraph.topological_ordering(),
                pgraph.outputs,
                [ppattern],
                incomings_fn=pgraph.node_inputs,
                outcomings_fn=pgraph.node_outputs,
                pattern_matcher_fn=_match_parameter_nodes_pattern,
            )
            if not matches:
                return None
            lpmatches[pname] = matches
        matched_parameters.append(lpmatches)

        # We got a match with the layer and its parameters.
        # Next, try to match its input sub-graph.
        matched_layers.append(layer)
        if lid != num_entries - 1:
            (layer,) = in_nodes

    return LayerOptMatch(pattern, matched_layers, matched_parameters)
