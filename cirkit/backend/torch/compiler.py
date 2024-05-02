import os
from collections import defaultdict
from typing import IO, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import Tensor

from cirkit.backend.base import AbstractCompiler, CompilerRegistry
from cirkit.backend.torch.layers import TorchInputLayer, TorchLayer
from cirkit.backend.torch.models import AbstractTorchCircuit, TorchCircuit, TorchConstantCircuit
from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.rules import (
    DEFAULT_LAYER_COMPILATION_RULES,
    DEFAULT_PARAMETER_COMPILATION_RULES,
)
from cirkit.backend.torch.semiring import Semiring, SemiringCls
from cirkit.backend.torch.utils import InitializerFunc
from cirkit.symbolic.circuit import Circuit, CircuitOperator, pipeline_topological_ordering
from cirkit.symbolic.layers import InputLayer, Layer, ProductLayer, SumLayer
from cirkit.symbolic.params import AbstractParameter


class TorchCompiler(AbstractCompiler):
    def __init__(self, semiring: str = "sum-product", fold: bool = False, einsum: bool = False):
        default_registry = CompilerRegistry(
            DEFAULT_LAYER_COMPILATION_RULES, DEFAULT_PARAMETER_COMPILATION_RULES
        )
        super().__init__(default_registry, fold=fold, einsum=einsum)
        self._compiled_layers: Dict[Layer, TorchLayer] = {}
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
            tc = self._compile_circuit(sci)

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

    def retrieve_parameter(self, layer: Layer, name: str) -> AbstractTorchParameter:
        compiled_layer = self._compiled_layers[layer]
        p = getattr(compiled_layer, name)
        if not isinstance(p, AbstractTorchParameter):
            raise ValueError(
                f"Attribute '{name}' of layer '{layer.__class__.__name__}' is not a parameter"
            )
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
        self, sc: Circuit, tc: AbstractTorchCircuit, compiled_layers: Dict[Layer, TorchLayer]
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

        # Construct the tensorized circuit, which is a torch module
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
        tc = tc_cls(
            sc.scope,
            sc.num_channels,
            layers=list(compiled_layers.values()),
            in_layers=in_layers,
            out_layers=out_layers,
        )

        # Apply optimizations
        tc, compiled_layers = self._optimize_circuit(tc, compiled_layers)

        # Register the compiled circuit, as well as the compiled layer infos
        self._register_compiled_circuit(sc, tc, compiled_layers)
        return tc

    def _optimize_circuit(
        self, tc: TorchCircuit, compiled_layers: Dict[Layer, TorchLayer]
    ) -> Tuple[TorchCircuit, Dict[Layer, TorchLayer]]:
        # Try to optimize the circuit by using einsums
        if self.is_einsum_enabled:
            tc = self._einsumize_circuit(tc)
        # Fold the circuit
        if self.is_fold_enabled:
            tc = self._fold_circuit(tc, compiled_layers)
        return tc, compiled_layers

    def _einsumize_circuit(self, tc: AbstractTorchCircuit) -> AbstractTorchCircuit:
        ...

    def _fold_circuit(
        self, tc: AbstractTorchCircuit, compiled_ls: Dict[Layer, TorchLayer]
    ) -> Tuple[AbstractTorchCircuit, Dict[Layer, TorchLayer]]:
        # The list of folded layers
        layers: List[TorchLayer] = []

        # A map from symbolic layer to folded compiled layers
        compiled_layers: Dict[Layer, TorchLayer] = {}

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

        # Fold layers in each inner frontier, by firstly finding the layer groups to fold
        # in each frontier, and then by stacking each group of layers into a folded layer
        for i, frontier in enumerate(frontiers_ordering):
            # Retrieve the layer groups we can fold
            layer_groups = self._group_foldable_layers(frontier)

            # Fold each group of layers
            for group in layer_groups:
                # Retrieve both the fold indices and within-fold index of the unfolded input layers
                in_layers: List[List[TorchLayer]] = [tc.layer_inputs(l) for l in group]

                # Check if we are folding input layers.
                # If that is the case, we index the data variables. We can still fold the layers
                # in such a group of input layers because they will be defined over the same
                # number of variables. If that is not the case, we retrieve the input index
                # from one of the useful maps.
                if i == 0:
                    in_layers_idx = [list(l.scope) for l in group]
                else:
                    in_layers_idx = [[fold_idx[li] for li in lsi] for lsi in in_layers]

                # Fold the layers group
                folded_layer = self._fold_layers_group(group)

                # Set the input and output folded layers
                flatten_in_layers = set(li for lsi in in_layers for li in lsi)
                folded_in_layers = [layers[fold_idx[l][0]] for l in flatten_in_layers]
                in_layers[folded_layer] = folded_in_layers
                for fl in folded_in_layers:
                    out_layers[fl].append(folded_layer)

                # Update the data structures
                layers.append(folded_layer)
                for i, l in enumerate(group):
                    fold_idx[l] = (len(layers), i)
                fold_in_layers_idx[folded_layer] = in_layers_idx

        # Similar as the 'fold_in_layers_idx' data structure above,
        # but indexing the entries of the folded outputs
        fold_out_layers_idx = [fold_idx[lo] for lo in tc.output_layers]

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
        return folded_tc, compiled_ls

    def _group_foldable_layers(self, layers: List[TorchLayer]) -> List[List[TorchLayer]]:
        ...

    def _fold_layers_group(self, layers: List[TorchLayer]) -> TorchLayer:
        ...

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
