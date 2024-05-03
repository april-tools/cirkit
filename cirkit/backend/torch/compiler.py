import os
from collections import defaultdict
from typing import IO, Dict, List, Optional, Tuple, Type, Union

import torch

from cirkit.backend.base import AbstractCompiler, CompilerRegistry
from cirkit.backend.torch.layers import TorchInnerLayer, TorchInputLayer, TorchLayer
from cirkit.backend.torch.models import AbstractTorchCircuit, TorchCircuit, TorchConstantCircuit
from cirkit.backend.torch.params import TorchParameter
from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.params.parameter import TorchConstantParameter
from cirkit.backend.torch.params.special import TorchFoldIdxParameter, TorchFoldParameter
from cirkit.backend.torch.rules import (
    DEFAULT_LAYER_COMPILATION_RULES,
    DEFAULT_PARAMETER_COMPILATION_RULES,
)
from cirkit.backend.torch.semiring import Semiring, SemiringCls
from cirkit.backend.torch.utils import InitializerFunc
from cirkit.symbolic.circuit import Circuit, CircuitOperator, pipeline_topological_ordering
from cirkit.symbolic.layers import Layer
from cirkit.symbolic.params import AbstractParameter


class TorchCompiler(AbstractCompiler):
    def __init__(self, semiring: str = "sum-product", fold: bool = False, einsum: bool = False):
        default_registry = CompilerRegistry(
            DEFAULT_LAYER_COMPILATION_RULES, DEFAULT_PARAMETER_COMPILATION_RULES
        )
        super().__init__(default_registry, fold=fold, einsum=einsum)
        self._compiled_layers: Dict[Layer, Tuple[TorchLayer, int]] = {}
        self._fold = fold
        self._einsum = einsum
        self._semiring = Semiring.from_name(semiring)

    def compile_pipeline(self, sc: Circuit) -> AbstractTorchCircuit:
        # Compile the circuits following the topological ordering of the pipeline.
        ordering = pipeline_topological_ordering({sc})
        for sci in ordering:
            # Check if the circuit in the pipeline has already been compiled
            if self.is_compiled(sci):
                continue

            # Compile the circuit
            self._compile_circuit(sci)

        # Return the compiled circuit (i.e., the output of the circuit pipeline)
        return self.get_compiled_circuit(sc)

    @property
    def is_fold_enabled(self) -> bool:
        return self._fold

    @property
    def is_einsum_enabled(self) -> bool:
        return self._einsum

    @property
    def semiring(self) -> SemiringCls:
        return self._semiring

    def retrieve_parameter(self, sl: Layer, name: str) -> AbstractTorchParameter:
        layer, layer_fold_idx = self._compiled_layers[sl]
        p = getattr(layer, name)
        if not isinstance(p, AbstractTorchParameter):
            raise ValueError(
                f"Attribute '{name}' of layer '{sl.__class__.__name__}' is not a parameter"
            )
        if self._fold:
            return TorchFoldIdxParameter(p, fold_idx=layer_fold_idx)
        return p

    def compile_parameter(
        self, parameter: AbstractParameter, *, init_func: Optional[InitializerFunc] = None
    ) -> AbstractTorchParameter:
        signature = type(parameter)
        func = self.retrieve_parameter_rule(signature)
        return func(self, parameter, init_func=init_func)

    def _compile_layer(self, layer: Layer) -> TorchLayer:
        signature = type(layer)
        func = self.retrieve_layer_rule(signature)
        return func(self, layer)

    def _register_compiled_circuit(
        self,
        sc: Circuit,
        tc: AbstractTorchCircuit,
        compiled_layers: Dict[Layer, Tuple[TorchLayer, int]],
    ):
        super().register_compiled_circuit(sc, tc)
        self._compiled_layers.update(compiled_layers)

    def _compile_circuit(self, sc: Circuit) -> AbstractTorchCircuit:
        # A map from symbolic to compiled layers
        compiled_layers: Dict[Layer, TorchLayer] = {}

        # The inputs and outputs for each layer
        in_layers: Dict[TorchLayer, List[TorchLayer]] = defaultdict(list)
        out_layers: Dict[TorchLayer, List[TorchLayer]] = defaultdict(list)

        # Compile layers by following the topological ordering
        for sl in sc.layers_topological_ordering():
            # Compile the layer, for any layer types
            layer = self._compile_layer(sl)

            # Build the connectivity between compiled layers
            ins = [compiled_layers[sli] for sli in sc.layer_inputs(sl)]
            in_layers[layer].extend(ins)
            for li in ins:
                out_layers[li].append(layer)
            compiled_layers[sl] = layer

        # If the symbolic circuit being compiled has been obtained by integrating
        # another circuit over all the variables it is defined on,
        # then return a 'constant circuit' whose interface does not require inputs
        if (
            sc.operation is not None
            and sc.operation.operator == CircuitOperator.INTEGRATION
            and sc.operation.metadata["scope"] == sc.scope
        ):
            tc_cls = TorchConstantCircuit
        else:
            tc_cls = TorchCircuit

        # Construct the tensorized circuit
        layers = list(compiled_layers.values())
        tc = tc_cls(
            sc.scope,
            sc.num_channels,
            layers=layers,
            in_layers=in_layers,
            out_layers=out_layers,
        )

        # Apply optimizations
        tc, fold_compiled_layers = self._optimize_circuit(tc, compiled_layers)

        # Register the compiled circuit, as well as the compiled layer infos
        self._register_compiled_circuit(sc, tc, fold_compiled_layers)
        return tc

    def _optimize_circuit(
        self, tc: TorchCircuit, compiled_layers: Dict[Layer, TorchLayer]
    ) -> Tuple[TorchCircuit, Dict[Layer, Tuple[TorchLayer, int]]]:
        # Try to optimize the circuit by using einsums
        if self.is_einsum_enabled:
            tc, einsum_compiled_layers = einsumize_circuit(self, tc, compiled_layers)
        else:
            einsum_compiled_layers = compiled_layers

        # Fold the circuit
        if self.is_fold_enabled:
            tc, fold_compiled_layers = fold_circuit(self, tc, einsum_compiled_layers)
        else:
            # Without folding, every compiled layer has fold dimension 1.
            # So, each symbolic layer gets mapped to the 0-slice of such 'improper' folded layer.
            fold_compiled_layers = {sl: (l, 0) for sl, l in einsum_compiled_layers.items()}

        # This function returns both the tensorized circuit and
        # a map from symbolic layer to (resp. folded) layers (resp. as well with the slice index)
        return tc, fold_compiled_layers

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


def einsumize_circuit(
    compiler: TorchCompiler, tc: AbstractTorchCircuit, compiled_layers: Dict[Layer, TorchLayer]
) -> Tuple[AbstractTorchCircuit, Dict[Layer, TorchLayer]]:
    ...


def fold_circuit(
    compiler: TorchCompiler, tc: AbstractTorchCircuit, compiled_layers: Dict[Layer, TorchLayer]
) -> Tuple[AbstractTorchCircuit, Dict[Layer, Tuple[TorchLayer, int]]]:
    # The list of folded layers
    layers: List[TorchLayer] = []

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

    # The inputs and outputs for each folded layer
    in_layers: Dict[TorchLayer, List[TorchLayer]] = {}
    out_layers: Dict[TorchLayer, List[TorchLayer]] = defaultdict(list)

    # Retrieve the layer-wise (aka bottom-up) topological ordering of layers
    frontiers_ordering: List[List[TorchLayer]] = tc.layerwise_topological_ordering()

    # Fold layers in each frontier, by firstly finding the layer groups to fold
    # in each frontier, and then by stacking each group of layers into a folded layer
    for i, frontier in enumerate(frontiers_ordering):
        # Retrieve the layer groups we can fold
        layer_groups = group_foldable_layers(compiler, frontier)

        # Fold each group of layers
        for group in layer_groups:
            # For each layer in the group, retrieve the unfolded input layers
            group_in_layers: List[List[TorchLayer]] = [tc.layer_inputs(l) for l in group]

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
            folded_layer = fold_layers_group(compiler, group)

            # Set the input and output folded layers
            folded_in_layers = list(
                set(layers[fold_idx[li][0]] for lsi in group_in_layers for li in lsi)
            )
            in_layers[folded_layer] = folded_in_layers
            for fl in folded_in_layers:
                out_layers[fl].append(folded_layer)

            # Update the data structures
            for j, l in enumerate(group):
                fold_idx[l] = (len(layers), j)
            layers.append(folded_layer)
            fold_in_layers_idx[folded_layer] = in_layers_idx

    # Similar as the 'fold_in_layers_idx' data structure above,
    # but indexing the entries of the folded outputs
    fold_out_layers_idx = [fold_idx[lo] for lo in tc.output_layers]

    # Construct a super useful map from a symbolic layer to a tuple
    # containing the corresponding folded layer and its slice index within that fold
    fold_compiled_layers: Dict[Layer, Tuple[TorchLayer, int]] = {
        sl: (layers[fold_idx[l][0]], fold_idx[l][1]) for sl, l in compiled_layers.items()
    }

    # Instantiate a folded circuit
    folded_tc = type(tc)(
        tc.scope,
        tc.num_channels,
        layers,
        in_layers,
        out_layers,
        fold_in_layers_idx=fold_in_layers_idx,
        fold_out_layers_idx=fold_out_layers_idx,
    )
    return folded_tc, fold_compiled_layers


def group_foldable_layers(
    compiler: TorchCompiler, frontier: List[TorchLayer]
) -> List[List[TorchLayer]]:
    # A dictionary mapping a layer configuration (see below),
    # which uniquely identifies a group of layers that can be folded,
    # into a group of layers.
    groups_map: Dict[Tuple[Type[TorchLayer], Tuple[int, ...]], List[TorchLayer]] = defaultdict(list)

    # For each layer, either create a new group or insert it into an existing one
    for l in frontier:
        if isinstance(l, TorchInputLayer):
            l_conf = type(l), (l.num_variables, l.num_channels, l.num_output_units)
        else:
            assert isinstance(l, TorchInnerLayer)
            l_conf = type(l), (l.num_input_units, l.num_output_units, l.arity)
        # Note that, if no suitable group has been found, this will introduce a new group
        groups_map[l_conf].append(l)
    groups = list(groups_map.values())
    return groups


def fold_layers_group(compiler: TorchCompiler, layers: List[TorchLayer]) -> TorchLayer:
    # Retrieve the class of the folded layer, as well as the configuration attributes
    fold_layer_cls = type(layers[0])
    fold_layer_conf = layers[0].config
    num_folds = len(layers)
    fold_layer_conf.update(num_folds=num_folds)

    # Retrieve the parameters of each layer
    layer_params: Dict[str, List[AbstractTorchParameter]] = defaultdict(list)
    for l in layers:
        lparams = l.params
        for n, p in lparams.items():
            layer_params[n].append(p)

    # Fold the parameters, if the layers has any
    if layer_params:
        # Just a check that we can fold the parameters, i.e.,
        # they must have the same length (aka the number of folds).
        assert set(len(ps) for n, ps in layer_params.items()) == {num_folds}

        # Fold each group of parameters
        fold_layer_params: Dict[str, AbstractTorchParameter] = {
            n: fold_parameters_group(compiler, ps) for n, ps in layer_params.items()
        }
    else:
        fold_layer_params: Dict[str, AbstractTorchParameter] = {}

    # Instantiate a new folded layer, using the folded layer configuration and the folded parameters
    fold_layer = fold_layer_cls(**fold_layer_conf, **fold_layer_params, semiring=compiler.semiring)
    return fold_layer


def fold_parameters_group(
    compiler: TorchCompiler, params: List[AbstractTorchParameter]
) -> AbstractTorchParameter:
    # TODO: we need a function like 'group_foldable_layers' above as to
    #       firstly find groups of parameters that can be folded, and then folds them.
    #       If we cannot fold some parameters we can still force the folding via slicing+concatenating tensors.
    #       The intuition about this choice is that the biggest cost at inference time will be the evaluation of
    #       the layers rather than the evaluation of the parameters. This is because the parameters are an order
    #       of magnitude smaller than the inputs/outputs of each layer, which instead have the batch dimension(s).
    #       In other words, it is ok to NOT being able to fold some parameter evaluations, if in the end we can still
    #       fold the layers.

    # Check all shapes (ignoring the fold dimension) match
    shapes = [p.shape for p in params]
    num_folds = len(params)
    assert len(set(shapes)) == 1

    fold_param_cls = type(params[0])
    if issubclass(fold_param_cls, (TorchParameter, TorchConstantParameter)):
        # Catch the case we are folding 'leaf' torch parameters
        assert all(p.requires_grad for p in params)
        fold_param = fold_param_cls(
            *shapes[0], num_folds=num_folds, requires_grad=params[0].requires_grad
        )
        fold_init_func = lambda t: t.copy_(torch.cat([p() for p in params]))
        fold_param.initialize(fold_init_func)
        return fold_param

    if issubclass(fold_param_cls, TorchFoldIdxParameter):
        # Catch the case were we are folding slices of other folded parameters
        # This case regularly fires when doing operations over folded circuits
        assert len(set(id(p.opd) for p in params)) == 1
        in_opd = params[0].opd
        fold_idx = [si for p in params for si in p.fold_idx]
        fold_idx = None if fold_idx == list(range(num_folds)) else fold_idx
        if fold_idx is None:
            return in_opd
        else:
            fold_param = TorchFoldParameter(in_opd, fold_idx=fold_idx)
        return fold_param

    # Retrieve the input parameters for every parameter
    in_params: Dict[str, List[AbstractTorchParameter]] = defaultdict(list)
    for p in params:
        pparams = p.params
        for n, in_p in pparams.items():
            in_params[n].append(in_p)

    # Fold the parameters, if the layers has any
    if in_params:
        # Just a check that we can fold the parameters, i.e.,
        # they must have the same length (aka the number of folds).
        assert set(len(in_ps) for n, in_ps in in_params.items()) == {num_folds}

        # Fold each group of input parameters
        fold_in_params: Dict[str, AbstractTorchParameter] = {
            n: fold_parameters_group(compiler, in_ps) for n, in_ps in in_params.items()
        }
    else:
        fold_in_params: Dict[str, AbstractTorchParameter] = {}

    # Build the folded parameter
    fold_param_conf = params[0].config
    fold_param = fold_param_cls(**fold_param_conf, **fold_in_params)
    return fold_param
