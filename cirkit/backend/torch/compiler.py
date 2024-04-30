import os
from typing import IO, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import Tensor

from cirkit.backend.base import AbstractCompiler, CompilerRegistry
from cirkit.backend.torch.layers import TorchLayer
from cirkit.backend.torch.models import (
    AbstractTensorizedCircuit,
    TensorizedCircuit,
    TensorizedConstantCircuit,
)
from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.rules import (
    DEFAULT_LAYER_COMPILATION_RULES,
    DEFAULT_PARAMETER_COMPILATION_RULES,
)
from cirkit.backend.torch.utils import InitializerFunc
from cirkit.symbolic.circuit import Circuit, CircuitOperator, pipeline_topological_ordering
from cirkit.symbolic.layers import InputLayer, Layer, ProductLayer, SumLayer
from cirkit.symbolic.params import AbstractParameter


class TorchCompiler(AbstractCompiler):
    def __init__(self, fold: bool = False, einsum: bool = False):
        default_registry = CompilerRegistry(
            DEFAULT_LAYER_COMPILATION_RULES, DEFAULT_PARAMETER_COMPILATION_RULES
        )
        super().__init__(default_registry, fold=fold, einsum=einsum)
        self._compiled_layers: Dict[Layer, TorchLayer] = {}

    def retrieve_parameter(self, layer: Layer, name: str) -> AbstractTorchParameter:
        compiled_layer = self._compiled_layers[layer]
        p = getattr(compiled_layer, name)
        if not isinstance(p, AbstractTorchParameter):
            raise ValueError(
                f"Attribute '{name}' of layer '{layer.__class__.__name__}' is not a parameter"
            )
        return p

    def compile_pipeline(self, sc: Circuit) -> AbstractTensorizedCircuit:
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

    def compile_parameter(
        self, parameter: AbstractParameter, *, init_func: Optional[InitializerFunc] = None
    ) -> AbstractTorchParameter:
        signature = type(parameter)
        func = self.retrieve_parameter_rule(signature)
        return func(self, parameter, init_func=init_func)

    def retrieve_initializer(self, layer_cls: Type[TorchLayer], name: str) -> InitializerFunc:
        return layer_cls.default_initializers()[name]

    def _register_compiled_circuit(
        self, sc: Circuit, tc: AbstractTensorizedCircuit, compiled_layer_ids: Dict[Layer, int]
    ):
        super().register_compiled_circuit(sc, tc)
        self._compiled_layers.update({l: tc.layers[i] for l, i in compiled_layer_ids.items()})

    def _compile_circuit(self, sc: Circuit) -> AbstractTensorizedCircuit:
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
        # If the symbolic circuit being compiled has been obtained by integrating
        # another circuit over all the variables it is defined on,
        # then return a 'constant circuit' whose interface does not require inputs
        if (
            sc.operation is not None
            and sc.operation.operator == CircuitOperator.INTEGRATION
            and sc.operation.metadata["scope"] == sc.scope
        ):
            circuit_cls = TensorizedConstantCircuit
        else:
            circuit_cls = TensorizedCircuit
        circuit = circuit_cls(sc.num_variables, sc.num_channels, layers, bookkeeping)

        # Register the compiled circuit, as well as the compiled layer infos
        self._register_compiled_circuit(sc, circuit, compiled_layer_ids)
        return circuit

    def _compile_layer(self, layer: Layer) -> TorchLayer:
        signature = type(layer)
        func = self.retrieve_layer_rule(signature)
        return func(self, layer)

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
