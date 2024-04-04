from typing import Dict, Type, Union

from cirkit.layers import Layer
from cirkit.symbolic.symb_circuit import SymbCircuit
from cirkit.symbolic.symb_layers import SymbLayer
from cirkit.symbolic.symb_params import AbstractSymbParameter
from cirkit.tensorized import TensorizedCircuit
from cirkit.tensorized.reparams import Reparameterization


class PipelineContext:
    def __init__(self, backend: str = "torch"):
        self.backend = backend
        self._named_symb: Dict[str, SymbCircuit] = {}
        self._symb_tensorized_map: Dict[SymbCircuit, TensorizedCircuit] = {}
        self._symb_layers_map: Dict[SymbCircuit, Dict[SymbLayer, int]] = {}

    def __getitem__(
        self, symb_circuit: Union[str, SymbCircuit]
    ) -> Union[SymbCircuit, TensorizedCircuit]:
        if isinstance(symb_circuit, str):
            return self._named_symb[symb_circuit]
        return self._symb_tensorized_map[symb_circuit]

    def __contains__(self, symb_circuit: SymbCircuit) -> bool:
        return symb_circuit in self._symb_tensorized_map

    def compile(self, symb_circuit: SymbCircuit, **opt_kwargs):
        if self.backend == "torch":
            # Compile using the torch backend
            from cirkit.tensorized.compilers.torch_module import compile_pipeline

            compile_pipeline({symb_circuit}, ctx=self, **opt_kwargs)
        else:
            raise NotImplementedError(f"Backend '{self.backend}' is not implemented")

    def save(self, symb_filepath: str, tens_filepath: str):
        pass

    @staticmethod
    def load(symb_filepath: str, tens_filepath: str) -> "PipelineContext":
        pass

    def register_layer_compilation_rule(self, symb_cls: Type[SymbLayer], layer_cls: Type[Layer]):
        pass

    def register_param_compilation_rule(
        self, symb_cls: Type[AbstractSymbParameter], param_cls: Type[Reparameterization]
    ):
        pass

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
