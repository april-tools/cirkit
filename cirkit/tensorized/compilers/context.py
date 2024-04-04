from typing import Dict

from cirkit.layers import Layer
from cirkit.symbolic.symb_circuit import SymbCircuit
from cirkit.symbolic.symb_layers import SymbLayer
from cirkit.tensorized import TensorizedCircuit


class PipelineContext:
    def __init__(self):
        self._symb_tensorized_map: Dict[SymbCircuit, TensorizedCircuit] = {}
        self._symb_layers_map: Dict[SymbCircuit, Dict[SymbLayer, int]] = {}

    def __getitem__(self, symb_circuit: SymbCircuit) -> TensorizedCircuit:
        return self._symb_tensorized_map[symb_circuit]

    def __contains__(self, symb_circuit: SymbCircuit) -> bool:
        return symb_circuit in self._symb_tensorized_map

    def compile(self, symb_circuit: SymbCircuit, backend: str = 'torch', **opt_kwargs):
        if backend == 'torch':
            # Compile using the torch backend
            from cirkit.tensorized.compilers.torch_module import compile_pipeline
            compile_pipeline({symb_circuit}, ctx=self, **opt_kwargs)
        else:
            raise NotImplementedError(f"Backend '{backend}' is not implemented")

    def _get_materialized_circuit(self, symb_circuit: SymbCircuit) -> TensorizedCircuit:
        return self[symb_circuit]

    def _get_materialized_layer(self, symb_circuit: SymbCircuit, symb_layer: SymbLayer) -> Layer:
        symb_layer_ids = self._symb_layers_map[symb_circuit]
        circuit = self[symb_circuit]
        return circuit.layers[symb_layer_ids[symb_layer]]

    def _update(self, ctx: "PipelineContext") -> "PipelineContext":
        self._symb_tensorized_map.update(ctx._symb_tensorized_map)
        self._symb_layers_map.update(ctx._symb_layers_map)
        return self

    def _register_materialized_circuit(
        self,
        symb_circuit: SymbCircuit,
        circuit: TensorizedCircuit,
        symb_layers_map: Dict[SymbLayer, int],
    ):
        self._symb_tensorized_map[symb_circuit] = circuit
        self._symb_layers_map[symb_circuit] = symb_layers_map
