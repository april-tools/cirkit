from typing import Dict, List, Optional, Tuple, Type, Union, cast

import torch
from torch import Tensor

from cirkit.backend.base import AbstractCompiler, CompilationRegistry
from cirkit.backend.torch.layers import (
    CategoricalLayer,
    ConstantLayer,
    DenseLayer,
    HadamardLayer,
    InnerLayer,
    InputLayer,
    KroneckerLayer,
    Layer,
    MixingLayer,
    SumLayer,
)
from cirkit.backend.torch.models import TensorizedCircuit
from cirkit.symbolic.sym_circuit import SymCircuit, pipeline_topological_ordering
from cirkit.symbolic.sym_layers import (
    AbstractSymLayerOperator,
    SymCategoricalLayer,
    SymConstantLayer,
    SymHadamardLayer,
    SymInputLayer,
    SymKroneckerLayer,
    SymLayer,
    SymMixingLayer,
    SymProdLayer,
    SymSumLayer,
)

_DEFAULT_COMPILATION_REGISTRY = CompilationRegistry(default_rules={})


class Compiler(AbstractCompiler):
    def __init__(self, **flags):
        super().__init__(_DEFAULT_COMPILATION_REGISTRY, **flags)
        self._symb_tensorized_map: Dict[SymCircuit, TensorizedCircuit] = {}
        self._symb_layers_map: Dict[SymCircuit, Dict[SymLayer, int]] = {}

    def compile(self, symb_circuit: SymCircuit):
        # Retrieve the topological ordering of the pipeline
        ordering = pipeline_topological_ordering({symb_circuit})

        # Materialize the circuits in the pipeline following the topological ordering
        # The parameters are saved in the input materialized circuits (for now),
        # and also shared across all the materialized circuits within the pipeline
        for symb_circuit in ordering:
            # Check if the circuit in the pipeline has already been materialized
            if symb_circuit in self:
                continue
            # Materialize the circuit
            self._compile_circuit(symb_circuit)

    def _compile_circuit(self, symb_circuit: SymCircuit):
        # The list of layers
        layers: List[Layer] = []

        # The bookkeeping data structure
        bookkeeping: List[Tuple[List[int], Optional[Tensor]]] = []

        # A useful map from symbolic layers to layer id (indices for the list of layers)
        symb_layers_map: Dict[SymLayer, int] = {}

        # Construct the bookkeeping data structure while compiling layers
        for sl in symb_circuit.layers_topological_ordering():
            if isinstance(sl, SymInputLayer):
                layer = self._compile_input_layer(symb_circuit, sl)
                bookkeeping_entry = ([], torch.tensor([list(sl.scope)]))
                bookkeeping.append(bookkeeping_entry)
            else:
                assert isinstance(sl, (SymSumLayer, SymProdLayer))
                layer = self._compile_inner_layer(symb_circuit, sl)
                bookkeeping_entry = ([symb_layers_map[isl] for isl in sl.inputs], None)
                bookkeeping.append(bookkeeping_entry)
            layer_id = len(layers)
            symb_layers_map[sl] = layer_id
            layers.append(layer)

        # Append a last bookkeeping entry with the info to extract the (possibly multiple) outputs
        output_indices = [symb_layers_map[sl] for sl in symb_circuit.output_layers]
        bookkeeping_entry = (output_indices, None)
        bookkeeping.append(bookkeeping_entry)

        # Construct the tensorized circuit object, and update the pipeline context
        circuit = TensorizedCircuit(symb_circuit, layers, bookkeeping)
        self._register_materialized_circuit(symb_circuit, circuit, symb_layers_map)

    def _compile_input_layer(
        self, symb_circuit: SymCircuit, symb_layer: SymInputLayer
    ) -> InputLayer:
        # Registry mapping symbolic input layers to executable layers classes
        materialize_input_registry: Dict[Type[SymInputLayer], Type[InputLayer]] = {
            SymConstantLayer: ConstantLayer,
            SymCategoricalLayer: CategoricalLayer,
        }

        layer_cls = materialize_input_registry[type(symb_layer)]

        symb_layer_operation = symb_layer.operation
        if symb_layer_operation is None:
            return layer_cls(
                num_input_units=symb_layer.num_channels,
                num_output_units=symb_layer.num_units,
                arity=len(symb_layer.scope),
                reparam=layer_cls.default_reparam(),
            )

        if symb_layer_operation.operator == AbstractSymLayerOperator.INTEGRATION:
            symb_circuit_op = symb_circuit.operation.operands[0]
            symb_layer_op = symb_layer_operation.operands[0]
            layer_op: InputLayer = cast(
                InputLayer, self._get_materialized_layer(symb_circuit_op, symb_layer_op)
            )
            return layer_cls(
                num_input_units=symb_layer.num_channels,
                num_output_units=symb_layer.num_units,
                arity=len(symb_layer.scope),
                reparam=layer_op.reparam,
            )

        assert False

    def _compile_inner_layer(
        self, symb_circuit: SymCircuit, symb_layer: Union[SymSumLayer, SymProdLayer]
    ) -> InnerLayer:
        # Registry mapping symbolic inner layers to executable layer classes
        materialize_inner_registry: Dict[Type[SymLayer], Type[InnerLayer]] = {
            SymSumLayer: DenseLayer,
            SymMixingLayer: MixingLayer,
            SymHadamardLayer: HadamardLayer,
            SymKroneckerLayer: KroneckerLayer,
        }

        layer_cls = materialize_inner_registry[type(symb_layer)]

        symb_layer_operation = symb_layer.operation
        if symb_layer_operation is None or not isinstance(symb_layer, SymSumLayer):
            return layer_cls(
                num_input_units=symb_layer.inputs[0].num_units,
                num_output_units=symb_layer.num_units,
                arity=len(symb_layer.inputs),
                **symb_layer.kwargs,
            )

        if symb_layer_operation.operator == AbstractSymLayerOperator.INTEGRATION:
            symb_circuit_op = symb_circuit.operation.operands[0]
            symb_layer_op = symb_layer_operation.operands[0]
            layer_op: SumLayer = cast(
                SumLayer, self._get_materialized_layer(symb_circuit_op, symb_layer_op)
            )
            return layer_cls(
                num_input_units=symb_layer.inputs[0].num_units,
                num_output_units=symb_layer.num_units,
                arity=len(symb_layer.inputs),
                reparam=layer_op.reparam,
            )

        assert False

    def __getitem__(self, symb_circuit: SymCircuit) -> TensorizedCircuit:
        return self._symb_tensorized_map[symb_circuit]

    def __contains__(self, symb_circuit: SymCircuit) -> bool:
        return symb_circuit in self._symb_tensorized_map

    def save(self, symb_filepath: str, tens_filepath: str):
        pass

    @staticmethod
    def load(symb_filepath: str, tens_filepath: str) -> "Compiler":
        pass

    def _get_materialized_circuit(self, symb_circuit: SymCircuit) -> TensorizedCircuit:
        return self[symb_circuit]

    def _get_materialized_layer(self, symb_circuit: SymCircuit, symb_layer: SymLayer) -> Layer:
        symb_layer_ids = self._symb_layers_map[symb_circuit]
        circuit = self[symb_circuit]
        return circuit.layers[symb_layer_ids[symb_layer]]

    def _register_materialized_circuit(
        self,
        symb_circuit: SymCircuit,
        circuit: TensorizedCircuit,
        symb_layers_map: Dict[SymLayer, int],
    ):
        self._symb_tensorized_map[symb_circuit] = circuit
        self._symb_layers_map[symb_circuit] = symb_layers_map
