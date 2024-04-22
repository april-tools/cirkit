from typing import Dict, List, Optional, Tuple, Type, Union, cast

import torch
from torch import Tensor

from cirkit.backend.base import AbstractCompiler, CompilerRegistry
from cirkit.backend.torch.layers import (
    CategoricalLayer,
    ConstantLayer,
    DenseLayer,
    HadamardLayer,
    InnerLayer,
    KroneckerLayer,
    MixingLayer,
    SumLayer,
    TorchInputLayer,
    TorchLayer,
)
from cirkit.backend.torch.models import TensorizedCircuit
from cirkit.backend.torch.rules import compile_dense_layer, compile_kronecker_layer
from cirkit.symbolic.circuit import Circuit, pipeline_topological_ordering
from cirkit.symbolic.layers import (
    AbstractLayerOperator,
    CategoricalLayer,
    ConstantLayer,
    DenseLayer,
    HadamardLayer,
    InputLayer,
    KroneckerLayer,
    Layer,
    MixingLayer,
    ProductLayer,
    SumLayer,
)
from cirkit.symbolic.params import AbstractParameter

_DEFAULT_COMPILATION_REGISTRY = CompilerRegistry(
    layer_rules={DenseLayer: compile_dense_layer, KroneckerLayer: compile_kronecker_layer},
    parameter_rules={},
)


class TorchCompiler(AbstractCompiler):
    def __init__(self, **flags):
        super().__init__(_DEFAULT_COMPILATION_REGISTRY, **flags)
        self._sym_layers_map: Dict[Circuit, Dict[TorchLayer, int]] = {}

    def compile_pipeline(self, sc: Circuit) -> TensorizedCircuit:
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

    def compile_learnable_parameter(self, parameter: AbstractParameter):
        pass

    def _register_compiled_circuit(
        self, sc: Circuit, tc: TensorizedCircuit, sym_layers_map: Dict[Layer, int]
    ):
        super().register_compiled_circuit(sc, tc)
        self._sym_layers_map[sc] = sym_layers_map

    def _compile_circuit(self, sym_circuit: Circuit) -> TensorizedCircuit:
        # The list of layers
        layers: List[TorchLayer] = []

        # The bookkeeping data structure
        bookkeeping: List[Tuple[List[int], Optional[Tensor]]] = []

        # A useful map from symbolic layers to layer id (indices for the list of layers)
        sym_layers_map: Dict[TorchLayer, int] = {}

        # Construct the bookkeeping data structure while compiling layers
        for sl in sym_circuit.layers_topological_ordering():
            if isinstance(sl, TorchInputLayer):
                layer = self._compile_input_layer(sym_circuit, sl)
                bookkeeping_entry = ([], torch.tensor([list(sl.scope)]))
                bookkeeping.append(bookkeeping_entry)
            else:
                assert isinstance(sl, (SumLayer, ProductLayer))
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

    def _compile_input_layer(self, sym_circuit: Circuit, sym_layer: InputLayer) -> InputLayer:
        # Registry mapping symbolic input layers to executable layers classes
        materialize_input_registry: Dict[Type[TorchInputLayer], Type[TorchInputLayer]] = {
            ConstantLayer: ConstantLayer,
            CategoricalLayer: CategoricalLayer,
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

        if sym_layer_operation.operator == AbstractLayerOperator.INTEGRATION:
            sym_circuit_op = sym_circuit.operation.operands[0]
            sym_layer_op = sym_layer_operation.operands[0]
            layer_op: TorchInputLayer = cast(
                TorchInputLayer, self._get_materialized_layer(sym_circuit_op, sym_layer_op)
            )
            return layer_cls(
                num_input_units=sym_layer.num_channels,
                num_output_units=sym_layer.num_units,
                arity=len(sym_layer.scope),
                reparam=layer_op.reparam,
            )

        assert False

    def _compile_inner_layer(
        self, sym_circuit: Circuit, sym_layer: Union[SumLayer, ProductLayer]
    ) -> InnerLayer:
        # Registry mapping symbolic inner layers to executable layer classes
        materialize_inner_registry: Dict[Type[TorchLayer], Type[InnerLayer]] = {
            SumLayer: DenseLayer,
            MixingLayer: MixingLayer,
            HadamardLayer: HadamardLayer,
            KroneckerLayer: KroneckerLayer,
        }

        layer_cls = materialize_inner_registry[type(sym_layer)]

        sym_layer_operation = sym_layer.operation
        if sym_layer_operation is None or not isinstance(sym_layer, SumLayer):
            return layer_cls(
                num_input_units=sym_layer.inputs[0].num_units,
                num_output_units=sym_layer.num_units,
                arity=len(sym_layer.inputs),
                **sym_layer.kwargs,
            )

        if sym_layer_operation.operator == AbstractLayerOperator.INTEGRATION:
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
