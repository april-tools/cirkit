import functools
import os
from collections import defaultdict
from typing import IO, Dict, List, Optional, Tuple, Union, cast

from torch import Tensor

from cirkit.backend.base import AbstractCompiler, CompilerRegistry
from cirkit.backend.torch.layers import TorchInnerLayer, TorchInputLayer, TorchLayer
from cirkit.backend.torch.models import AbstractTorchCircuit, TorchCircuit, TorchConstantCircuit
from cirkit.backend.torch.parameters.graph import (
    TorchParameter,
    TorchParameterNode,
    TorchParameterOp,
)
from cirkit.backend.torch.parameters.leaves import TorchPointerParameter, TorchTensorParameter
from cirkit.backend.torch.rules import (
    DEFAULT_LAYER_COMPILATION_RULES,
    DEFAULT_PARAMETER_COMPILATION_RULES,
)
from cirkit.backend.torch.semiring import Semiring, SemiringCls
from cirkit.symbolic.circuit import Circuit, CircuitOperator, pipeline_topological_ordering
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
    def __init__(self, semiring: str = "sum-product", fold: bool = False, einsum: bool = False):
        default_registry = CompilerRegistry(
            DEFAULT_LAYER_COMPILATION_RULES, DEFAULT_PARAMETER_COMPILATION_RULES
        )
        super().__init__(default_registry, fold=fold, einsum=einsum)
        self._fold = fold
        self._einsum = einsum
        self._semiring = Semiring.from_name(semiring)

        # The state of the compiler
        self._state = TorchCompilerState()

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
    def semiring(self) -> SemiringCls:
        return self._semiring

    @property
    def is_fold_enabled(self) -> bool:
        return self._fold

    @property
    def is_einsum_enabled(self) -> bool:
        return self._einsum

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
            in_compiled_nodes = [
                compiled_nodes_map[pi] for pi in parameter.node_inputs(p)
            ]
            in_nodes[compiled_p] = in_compiled_nodes
            for pi in in_compiled_nodes:
                out_nodes[pi].append(compiled_p)
            compiled_nodes_map[p] = compiled_p
            nodes.append(compiled_p)

        # Build the parameter's computational graph
        return TorchParameter(nodes, in_nodes, out_nodes, topologically_ordered=True)

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
            topologically_ordered=True
        )

        # Apply optimizations
        cc = self._optimize_circuit(cc)

        # Initialize the circuit parameters
        _initialize_circuit_parameters(self, cc)

        # Register the compiled circuit, as well as the compiled layer infos
        self.register_compiled_circuit(sc, cc)

        # Signal the end of the circuit compilation to the state
        self._state.finish_compilation()
        return cc

    def _optimize_circuit(self, cc: AbstractTorchCircuit) -> AbstractTorchCircuit:
        if self.is_einsum_enabled:
            # Optimize the circuit by using einsums
            opt_cc = _einsumize_circuit(self, cc)
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


def _initialize_circuit_parameters(compiler: TorchCompiler, cc: AbstractTorchCircuit) -> None:
    # For each layer, initialize its parameters, if any
    for l in cc.layers:
        for _, p in l.params.items():
            p.initialize_()


def _einsumize_circuit(compiler: TorchCompiler, cc: AbstractTorchCircuit) -> AbstractTorchCircuit:
    ...


def _fold_circuit(compiler: TorchCompiler, cc: AbstractTorchCircuit) -> AbstractTorchCircuit:
    # A useful data structure mapping each unfolded layer to
    # (i) a 'fold_id' (a natural number) pointing to the folded layer it is associated to; and
    # (ii) a 'slice_idx' (a natural number) within the output of the folded layer,
    #      which recovers the output of the unfolded layer.
    fold_idx: Dict[TorchLayer, Tuple[int, int]] = {}

    # A useful data structure mapping each folded layer to
    # a tensor of indices IDX of size (F, H, 2), where F is the number of layers in the fold,
    # H is the number of inputs to each fold. Each entry i,j,: of IDX is a pair (fold_id, slice_idx),
    # pointing to the folded layer of id 'fold_id' and to the slice 'slice_idx' within that fold.
    fold_in_layers_idx: Dict[TorchLayer, List[List[Tuple[int, int]]]] = {}

    # The list of folded layers
    layers: List[TorchLayer] = []

    # The inputs and outputs for each folded layer
    in_layers: Dict[TorchLayer, List[TorchLayer]] = {}
    out_layers: Dict[TorchLayer, List[TorchLayer]] = defaultdict(list)

    # Fold layers in each frontier, by firstly finding the layer groups to fold
    # in each frontier, and then by stacking each group of layers into a folded layer
    for frontier in cc.layerwise_topological_ordering():
        # Retrieve the layer groups we can fold
        layer_groups = _group_foldable_layers(frontier)

        # Fold each group of layers
        for group in layer_groups:
            # For each layer in the group, retrieve the unfolded input layers
            group_in_layers: List[List[TorchLayer]] = [cc.layer_inputs(l) for l in group]

            # Check if we are folding input layers.
            # If that is the case, we index the data variables. We can still fold the layers
            # in such a group of input layers because they will be defined over the same
            # number of variables. If that is not the case, we retrieve the input index
            # from one of the useful maps.
            in_layers_idx: List[List[Tuple[int, int]]]
            is_folding_input = len(group_in_layers[0]) == 0
            if is_folding_input:
                in_layers_idx = [[(-1, s) for s in l.scope] for l in group]
            else:
                in_layers_idx = [[fold_idx[li] for li in lsi] for lsi in group_in_layers]

            # Fold the layers group
            folded_layer = _fold_layers_group(compiler, group)

            # Set the input and output folded layers
            folded_in_layers = list(
                set(layers[fold_idx[li][0]] for lsi in group_in_layers for li in lsi)
            )
            in_layers[folded_layer] = folded_in_layers
            for fl in folded_in_layers:
                out_layers[fl].append(folded_layer)

            # Update the data structures
            for i, l in enumerate(group):
                fold_idx[l] = (len(layers), i)
            layers.append(folded_layer)
            fold_in_layers_idx[folded_layer] = in_layers_idx

    # Similar as the 'fold_in_layers_idx' data structure above,
    # but indexing the entries of the folded outputs
    fold_out_layers_idx = [fold_idx[lo] for lo in cc.outputs]

    # Instantiate a folded circuit
    folded_cc = type(cc)(
        cc.scope,
        cc.num_channels,
        layers,
        in_layers,
        out_layers,
        topologically_ordered=True,
        fold_in_layers_idx=fold_in_layers_idx,
        fold_out_layers_idx=fold_out_layers_idx,
    )
    return folded_cc


def _fold_layers_group(compiler: TorchCompiler, layers: List[TorchLayer]) -> TorchLayer:
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
    fold_layer = fold_layer_cls(
        **fold_layer_conf, **fold_layer_parameters, semiring=compiler.semiring
    )
    return fold_layer


def _group_foldable_layers(frontier: List[TorchLayer]) -> List[List[TorchLayer]]:
    # A dictionary mapping a layer configuration (see below),
    # which uniquely identifies a group of layers that can be folded,
    # into a group of layers.
    groups_map: Dict[tuple, List[TorchLayer]] = defaultdict(list)

    # For each layer, either create a new group or insert it into an existing one
    for l in frontier:
        if isinstance(l, TorchInputLayer):
            l_conf = type(l), l.num_variables, l.num_channels, l.num_output_units
        else:
            assert isinstance(l, TorchInnerLayer)
            l_conf = type(l), l.num_input_units, l.num_output_units, l.arity
        # Note that, if no suitable group has been found, this will introduce a new group
        groups_map[l_conf].append(l)
    return list(groups_map.values())


def _fold_parameters(compiler: TorchCompiler, parameters: List[TorchParameter]) -> TorchParameter:
    # Retrieve:
    # (i)  the parameter nodes and the output shapes for each of them;
    # (ii) the layer-wise (aka bottom-up) topological orderings of parameter nodes
    in_nodes: Dict[TorchParameterNode, List[TorchParameterNode]] = {}
    frontiers_ordering: List[List[TorchParameterNode]] = []
    for pi in parameters:
        in_nodes.update(pi.nodes_inputs)
        for i, frontier in enumerate(pi.layerwise_topological_ordering()):
            if i < len(frontiers_ordering):
                frontiers_ordering[i].extend(frontier)
                continue
            frontiers_ordering.append(frontier)

    # A useful data structure mapping each unfolded parameter node to
    # (i) a 'fold_id' (a natural number) pointing to the folded parameter node it is associated to; and
    # (ii) a 'slice_idx' (a natural number) within the output of the parameter node,
    #      which recovers the output of the unfolded parameter node.
    fold_idx: Dict[TorchParameterNode, Tuple[int, int]] = {}

    # A useful data structure mapping each folded parameter node to
    # a list of F indices IDX, each of size (F, H'), where H' is the number of inputs to that fold.
    # Each entry i,j,: of IDX is a pair (fold_id, slice_idx), pointing to the folded parameter node
    # of id 'fold_id' and to the slice 'slice_idx' within that fold.
    # Note that IDX can be possibly optional. If that is the case, then the folded parameter node is
    # actually a folded parameter leaf.
    fold_in_nodes_idx: Dict[TorchParameterNode, Optional[List[List[Tuple[int, int]]]]] = {}

    # The list of folded parameter nodes
    folded_nodes: List[TorchParameterNode] = []

    # The inputs and outputs for each folded parameter node
    in_folded_nodes: Dict[TorchParameterNode, List[TorchParameterNode]] = {}
    out_folded_nodes: Dict[TorchParameterNode, List[TorchParameterNode]] = defaultdict(list)

    # Fold parameter nodes in each frontier, by firstly finding the parameter node groups to fold
    # in each frontier, and then by stacking each group of parameter nodes into a folded parameter node
    for frontier in frontiers_ordering:
        # Retrieve the parameter node groups we can fold
        parameter_groups = _group_foldable_parameter_nodes(frontier, in_nodes)

        # Fold each group of parameter nodes
        for group in parameter_groups:
            # For each parameter node in the group, retrieve the unfolded input nodes
            group_in_nodes: List[List[TorchParameterNode]] = [in_nodes.get(p, []) for p in group]

            # Check if we are folding input layers
            in_nodes_idx: Optional[List[List[Tuple[int, int]]]]
            is_folding_input = len(group_in_nodes[0]) == 0
            if is_folding_input:
                in_nodes_idx = None
            else:
                in_nodes_idx = [[fold_idx[pi] for pi in psi] for psi in group_in_nodes]

            # Fold the parameter nodes
            folded_node = _fold_parameter_nodes_group(compiler, group)

            # Set the input and output folded nodes
            folded_in_nodes = list(
                set(folded_nodes[fold_idx[pi][0]] for psi in group_in_nodes for pi in psi)
            )
            in_folded_nodes[folded_node] = folded_in_nodes
            for fp in folded_in_nodes:
                out_folded_nodes[fp].append(folded_node)

            # Update the data structures
            for i, p in enumerate(group):
                fold_idx[p] = (len(folded_nodes), i)
            folded_nodes.append(folded_node)
            fold_in_nodes_idx[folded_node] = in_nodes_idx

    # Similar as the 'fold_in_nodes_idx' data structure above,
    # but indexing the entries of the folded outputs.
    # This is needed as to retrieve the final folded output, regardless
    # of how the frontiers have been folded.
    fold_out_nodes_idx = [fold_idx[pi.output] for pi in parameters]

    # Construct the folded parameter's computational graph
    folded_parameter = TorchParameter(
        folded_nodes,
        in_folded_nodes,
        out_folded_nodes,
        topologically_ordered=True,
        fold_in_nodes_idx=fold_in_nodes_idx,
        fold_out_nodes_idx=fold_out_nodes_idx,
    )
    return folded_parameter


def _fold_init_tensors(t: Tensor, *, ps: List[TorchTensorParameter]) -> Tensor:
    # Initialize a folded tensor by using the initializers of the tensor parameter nodes being folded
    for i, p in enumerate(ps):
        init_func = p.init_func
        if init_func is not None:
            init_func(t[i])
    return t


def _fold_parameter_nodes_group(
    compiler: TorchCompiler, group: List[TorchParameterNode]
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
            init_func=functools.partial(_fold_init_tensors, ps=group),
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
        assert all(len(p.fold_idx) == 1 for p in group)
        in_folded_node = group[0].deref()
        in_fold_idx: List[int] = [p.fold_idx[0] for p in group]
        fold_idx = None if in_fold_idx == list(range(in_folded_node.num_folds)) else in_fold_idx
        return TorchPointerParameter(in_folded_node, fold_idx=fold_idx)
    # We are folding an operator: just set the number of folds and copy the configuration parameters
    assert all(isinstance(p, TorchParameterOp) for p in group)
    return fold_node_cls(*group[0].in_shapes, num_folds=len(group), **group[0].config)


def _group_foldable_parameter_nodes(
    nodes: List[TorchParameterNode], in_nodes: Dict[TorchParameterNode, List[TorchParameterNode]]
) -> List[List[TorchParameterNode]]:
    # A dictionary mapping a parameter configuration (see below),
    # which uniquely identifies a group of parameters that can be folded,
    # into a group of parameters.
    groups_map: Dict[tuple, List[TorchParameterNode]] = defaultdict(list)

    # For each parameter node, either create a new group or insert it into an existing one
    for i, p in enumerate(nodes):
        if isinstance(p, TorchTensorParameter):
            p_conf = type(p), p.shape, p.requires_grad
        elif isinstance(p, TorchPointerParameter):
            # We fold pointers only if the point to the same parameter tensor
            p_conf = type(p), p.shape, id(p.deref())
        else:
            assert isinstance(p, TorchParameterOp)
            # We can fold non-leaf parameter nodes only if they have inputs of the same shape.
            # This implies that the outputs shapes are equal.
            # Also, they must have equal configuration.
            in_shapes = p.in_shapes
            config = tuple(p.config.items())
            p_conf = type(p), *config, *in_shapes

        # Note that, if no suitable group has been found, this will introduce a new group
        groups_map[p_conf].append(p)
    return list(groups_map.values())
