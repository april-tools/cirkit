import os
from typing import Dict, List, Optional, Tuple, Type, Union, cast, Any, IO

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
    SymSumLayer
)
from cirkit.symbolic.sym_params import AbstractSymParameter

_DEFAULT_COMPILATION_REGISTRY = CompilationRegistry(default_rules={})


class TorchCompiler(AbstractCompiler):
    def __init__(self, **flags):
        super().__init__(_DEFAULT_COMPILATION_REGISTRY, **flags)
        self._sym_layers_map: Dict[SymCircuit, Dict[SymLayer, int]] = {}

    def compile_pipeline(self, sc: SymCircuit) -> TensorizedCircuit:
        # Retrieve the topological ordering of the pipeline
        ordering = pipeline_topological_ordering({sc})

        # Materialize the circuits in the pipeline following the topological ordering
        # The parameters are saved in the input materialized circuits (for now),
        # and also shared across all the materialized circuits within the pipeline
        for sci in ordering:
            # Check if the circuit in the pipeline has already been materialized
            if sci in self:
                continue
            # Materialize the circuit
            self._compile_circuit(sci)
        return self.get_compiled_circuit(sc)

    def compile_learnable_parameter(self, sym_param: AbstractSymParameter):
        pass

    def _register_compiled_circuit(
            self,
            sc: SymCircuit,
            tc: TensorizedCircuit,
            sym_layers_map: Dict[SymLayer, int]
    ):
        super().register_compiled_circuit(sc, tc)
        self._sym_layers_map[sc] = sym_layers_map

    def _compile_circuit(self, sym_circuit: SymCircuit) -> TensorizedCircuit:
        # The list of layers
        layers: List[Layer] = []

        # The bookkeeping data structure
        bookkeeping: List[Tuple[List[int], Optional[Tensor]]] = []

        # A useful map from symbolic layers to layer id (indices for the list of layers)
        sym_layers_map: Dict[SymLayer, int] = {}

        # Construct the bookkeeping data structure while compiling layers
        for sl in sym_circuit.layers_topological_ordering():
            if isinstance(sl, SymInputLayer):
                layer = self._compile_input_layer(sym_circuit, sl)
                bookkeeping_entry = ([], torch.tensor([list(sl.scope)]))
                bookkeeping.append(bookkeeping_entry)
            else:
                assert isinstance(sl, (SymSumLayer, SymProdLayer))
                layer = self._compile_inner_layer(sym_circuit, sl)
                bookkeeping_entry = ([sym_layers_map[isl] for isl in sl.inputs], None)
                bookkeeping.append(bookkeeping_entry)
            layer_id = len(layers)
            sym_layers_map[sl] = layer_id
            layers.append(layer)

        # Append a last bookkeeping entry with the info to extract the (possibly multiple) outputs
        output_indices = [sym_layers_map[sl] for sl in sym_circuit.output_layers]
        bookkeeping_entry = (output_indices, None)
        bookkeeping.append(bookkeeping_entry)

        # Construct the tensorized circuit object, and update the pipeline context
        circuit = TensorizedCircuit(sym_circuit, layers, bookkeeping)
        self._register_compiled_circuit(sym_circuit, circuit, sym_layers_map)
        return circuit

    def _compile_input_layer(
        self, sym_circuit: SymCircuit, sym_layer: SymInputLayer
    ) -> InputLayer:
        # Registry mapping symbolic input layers to executable layers classes
        materialize_input_registry: Dict[Type[SymInputLayer], Type[InputLayer]] = {
            SymConstantLayer: ConstantLayer,
            SymCategoricalLayer: CategoricalLayer,
        }

        layer_cls = materialize_input_registry[type(sym_layer)]

        sym_layer_operation = sym_layer.operation
        if sym_layer_operation is None:
            return layer_cls(
                num_input_units=sym_layer.num_channels,
                num_output_units=sym_layer.num_units,
                arity=len(sym_layer.scope),
                reparam=layer_cls.default_reparam(),
            )

        if sym_layer_operation.operator == AbstractSymLayerOperator.INTEGRATION:
            sym_circuit_op = sym_circuit.operation.operands[0]
            sym_layer_op = sym_layer_operation.operands[0]
            layer_op: InputLayer = cast(
                InputLayer, self._get_materialized_layer(sym_circuit_op, sym_layer_op)
            )
            return layer_cls(
                num_input_units=sym_layer.num_channels,
                num_output_units=sym_layer.num_units,
                arity=len(sym_layer.scope),
                reparam=layer_op.reparam,
            )

        assert False

    def _compile_inner_layer(
        self, sym_circuit: SymCircuit, sym_layer: Union[SymSumLayer, SymProdLayer]
    ) -> InnerLayer:
        # Registry mapping symbolic inner layers to executable layer classes
        materialize_inner_registry: Dict[Type[SymLayer], Type[InnerLayer]] = {
            SymSumLayer: DenseLayer,
            SymMixingLayer: MixingLayer,
            SymHadamardLayer: HadamardLayer,
            SymKroneckerLayer: KroneckerLayer,
        }

        layer_cls = materialize_inner_registry[type(sym_layer)]

        sym_layer_operation = sym_layer.operation
        if sym_layer_operation is None or not isinstance(sym_layer, SymSumLayer):
            return layer_cls(
                num_input_units=sym_layer.inputs[0].num_units,
                num_output_units=sym_layer.num_units,
                arity=len(sym_layer.inputs),
                **sym_layer.kwargs,
            )

        if sym_layer_operation.operator == AbstractSymLayerOperator.INTEGRATION:
            sym_circuit_op = sym_circuit.operation.operands[0]
            sym_layer_op = sym_layer_operation.operands[0]
            layer_op: SumLayer = cast(
                SumLayer, self._get_materialized_layer(sym_circuit_op, sym_layer_op)
            )
            return layer_cls(
                num_input_units=sym_layer.inputs[0].num_units,
                num_output_units=sym_layer.num_units,
                arity=len(sym_layer.inputs),
                reparam=layer_op.reparam,
            )

        assert False
