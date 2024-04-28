import os
from typing import IO, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from cirkit.backend.base import (
    AbstractCompiler,
    CompilerRegistry,
    LayerCompilationFunc,
    LayerCompilationSign,
    ParameterCompilationFunc,
    ParameterCompilationSign,
)
from cirkit.backend.torch.layers import TorchLayer
from cirkit.backend.torch.models import TensorizedCircuit
from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.rules import (
    compile_categorical_layer,
    compile_dense_layer,
    compile_gaussian_layer,
    compile_hadamard_layer,
    compile_kronecker_layer,
    compile_mixing_layer,
    compile_parameter,
    compile_placeholder_parameter,
)
from cirkit.symbolic.circuit import Circuit, pipeline_topological_ordering
from cirkit.symbolic.layers import (
    CategoricalLayer,
    DenseLayer,
    GaussianLayer,
    HadamardLayer,
    InputLayer,
    KroneckerLayer,
    Layer,
    MixingLayer,
    PlaceholderParameter,
    ProductLayer,
    SumLayer,
)
from cirkit.symbolic.params import AbstractParameter, Parameter

_DEFAULT_LAYER_COMPILATION_RULES: Dict[LayerCompilationSign, LayerCompilationFunc] = {
    CategoricalLayer: compile_categorical_layer,
    GaussianLayer: compile_gaussian_layer,
    HadamardLayer: compile_hadamard_layer,
    KroneckerLayer: compile_kronecker_layer,
    DenseLayer: compile_dense_layer,
    MixingLayer: compile_mixing_layer,
}

_DEFAULT_PARAMETER_COMPILATION_RULES: Dict[ParameterCompilationSign, ParameterCompilationFunc] = {
    Parameter: compile_parameter,
    PlaceholderParameter: compile_placeholder_parameter,
}


class TorchCompiler(AbstractCompiler):
    def __init__(self, **flags):
        default_registry = CompilerRegistry(
            _DEFAULT_LAYER_COMPILATION_RULES, _DEFAULT_PARAMETER_COMPILATION_RULES
        )
        super().__init__(default_registry, **flags)
        self._compiled_layers: Dict[Layer, TorchLayer] = {}

    def retrieve_parameter(self, layer: Layer, name: str) -> AbstractTorchParameter:
        compiled_layer = self._compiled_layers[layer]
        p = getattr(compiled_layer, name)
        if not isinstance(p, AbstractTorchParameter):
            raise ValueError(
                f"Attribute '{name}' of layer '{layer.__class__.__name__}' is not a parameter"
            )
        return p

    def compile_pipeline(self, sc: Circuit) -> TensorizedCircuit:
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

    def compile_parameter(self, parameter: AbstractParameter):
        signature = type(parameter)
        func = self.retrieve_parameter_rule(signature)
        return func(parameter, self)

    def _register_compiled_circuit(
        self, sc: Circuit, tc: TensorizedCircuit, compiled_layer_ids: Dict[Layer, int]
    ):
        super().register_compiled_circuit(sc, tc)
        self._compiled_layers.update({l: tc.layers[i] for l, i in compiled_layer_ids.items()})

    def _compile_circuit(self, sc: Circuit) -> TensorizedCircuit:
        # The list of layers
        layers: List[TorchLayer] = []

        # The bookkeeping data structure
        bookkeeping: List[Tuple[List[int], Optional[Tensor]]] = []

        # A useful map from layers to compiled layer ids (i.e., indices to the list of layers)
        compiled_layer_ids: Dict[Layer, int] = {}

        # Compile layers by following the topological ordering, while constructing the bookkeeping
        for sl in sc.layers_topological_ordering():
            # Compile the layer, for any layer types
            layer = self._compile_layer(sl)

            if isinstance(sl, InputLayer):
                # For input layers, the bookkeeping entry is a tensor index to the input tensor
                bookkeeping_entry = ([], torch.tensor([list(sl.scope)]))
            else:
                # For sum/product layers, the bookkeeping entry consists of the indices to the layer inputs
                assert isinstance(sl, (SumLayer, ProductLayer))
                bookkeeping_entry = ([compiled_layer_ids[sli] for sli in sc.layer_inputs(sl)], None)
            bookkeeping.append(bookkeeping_entry)
            compiled_layer_ids[sl] = len(layers)
            layers.append(layer)

        # Append a last bookkeeping entry with the info to extract the output tensor
        # This is necessary because we might have circuits with multiple outputs
        output_ids = [compiled_layer_ids[slo] for slo in sc.output_layers]
        bookkeeping_entry = (output_ids, None)
        bookkeeping.append(bookkeeping_entry)

        # Construct the tensorized circuit, which is a torch module
        circuit = TensorizedCircuit(layers, bookkeeping)

        # Register the compiled circuit, as well as the compiled layer infos
        self._register_compiled_circuit(sc, circuit, compiled_layer_ids)
        return circuit

    def _compile_layer(self, layer: Layer) -> TorchLayer:
        signature = type(layer)
        func = self.retrieve_layer_rule(signature)
        return func(layer, self)

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
