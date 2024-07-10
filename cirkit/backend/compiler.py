import os
from abc import ABC, abstractmethod
from typing import IO, Any, Optional, Protocol, Type, TypeVar, Union

from cirkit.backend.registry import CompilerRegistry
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import Initializer
from cirkit.symbolic.layers import Layer
from cirkit.symbolic.parameters import ParameterNode
from cirkit.utils.algorithms import BiMap

SUPPORTED_BACKENDS = ["torch"]

CompiledCircuit = TypeVar("CompiledCircuit")


class CompiledCircuitsMap:
    def __init__(self):
        self._bimap = BiMap[Circuit, CompiledCircuit]()

    def is_compiled(self, sc: Circuit) -> bool:
        return self._bimap.has_left(sc)

    def has_symbolic(self, cc: CompiledCircuit) -> bool:
        return self._bimap.has_right(cc)

    def get_compiled_circuit(self, sc: Circuit) -> CompiledCircuit:
        return self._bimap.get_left(sc)

    def get_symbolic_circuit(self, cc: CompiledCircuit) -> Circuit:
        return self._bimap.get_right(cc)

    def register_compiled_circuit(self, sc: Circuit, cc: CompiledCircuit):
        self._bimap.add(sc, cc)


LayerCompilationSign = Type[Layer]
ParameterCompilationSign = Type[ParameterNode]
InitializerCompilationSign = Type[Initializer]


class LayerCompilationFunc(Protocol):
    def __call__(self, compiler: "AbstractCompiler", sl: Layer, **kwargs) -> Any:
        ...


class ParameterCompilationFunc(Protocol):
    def __call__(self, compiler: "AbstractCompiler", p: ParameterNode, **kwargs) -> Any:
        ...


class InitializerCompilationFunc(Protocol):
    def __call__(self, compiler: "AbstractCompiler", init: Initializer, **kwargs) -> Any:
        ...


class CompilationRuleNotFound(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class CompilerLayerRegistry(CompilerRegistry[LayerCompilationSign, LayerCompilationFunc]):
    @classmethod
    def _validate_rule_signature(cls, func: LayerCompilationFunc) -> Optional[LayerCompilationSign]:
        args = func.__annotations__
        if "return" not in args or "compiler" not in args or len(args) != 3:
            return None
        if not issubclass(args["compiler"], AbstractCompiler):
            return None
        arg_names = list(filter(lambda a: a not in ("return", "compiler"), args.keys()))
        found_sym_cls = args[arg_names[0]]
        if not issubclass(found_sym_cls, Layer):
            return None
        return found_sym_cls


class CompilerParameterRegistry(
    CompilerRegistry[ParameterCompilationSign, ParameterCompilationFunc]
):
    @classmethod
    def _validate_rule_signature(
        cls, func: ParameterCompilationFunc
    ) -> Optional[ParameterCompilationSign]:
        args = func.__annotations__
        if "return" not in args or "compiler" not in args or len(args) != 3:
            return None
        if not issubclass(args["compiler"], AbstractCompiler):
            return None
        arg_names = list(filter(lambda a: a not in ("return", "compiler"), args.keys()))
        found_sym_cls = args[arg_names[0]]
        if not issubclass(found_sym_cls, ParameterNode):
            return None
        return found_sym_cls


class CompilerInitializerRegistry(
    CompilerRegistry[InitializerCompilationSign, InitializerCompilationFunc]
):
    @classmethod
    def _validate_rule_signature(
        cls, func: ParameterCompilationFunc
    ) -> Optional[InitializerCompilationSign]:
        args = func.__annotations__
        if "return" not in args or "compiler" not in args or len(args) != 3:
            return None
        if not issubclass(args["compiler"], AbstractCompiler):
            return None
        arg_names = list(filter(lambda a: a not in ("return", "compiler"), args.keys()))
        found_sym_cls = args[arg_names[0]]
        if not issubclass(found_sym_cls, Initializer):
            return None
        return found_sym_cls


class AbstractCompiler(ABC):
    def __init__(
        self,
        layers_registry: CompilerLayerRegistry,
        parameters_registry: CompilerParameterRegistry,
        initializers_registry: CompilerInitializerRegistry,
        **flags,
    ):
        self._layers_registry = layers_registry
        self._parameters_registry = parameters_registry
        self._initializers_registry = initializers_registry
        self._flags = flags
        self._compiled_circuits = CompiledCircuitsMap()

    def is_compiled(self, sc: Circuit) -> bool:
        return self._compiled_circuits.is_compiled(sc)

    def has_symbolic(self, cc: CompiledCircuit) -> bool:
        return self._compiled_circuits.has_symbolic(cc)

    def get_compiled_circuit(self, sc: Circuit) -> CompiledCircuit:
        return self._compiled_circuits.get_compiled_circuit(sc)

    def get_symbolic_circuit(self, cc: CompiledCircuit) -> Circuit:
        return self._compiled_circuits.get_symbolic_circuit(cc)

    def register_compiled_circuit(self, sc: Circuit, cc: CompiledCircuit):
        self._compiled_circuits.register_compiled_circuit(sc, cc)

    def add_layer_rule(self, func: LayerCompilationFunc):
        self._layers_registry.add_rule(func)

    def add_parameter_rule(self, func: ParameterCompilationFunc):
        self._parameters_registry.add_rule(func)

    def add_initializer_rule(self, func: InitializerCompilationFunc):
        self._initializers_registry.add_rule(func)

    def retrieve_layer_rule(self, signature: LayerCompilationSign) -> LayerCompilationFunc:
        return self._layers_registry.retrieve_rule(signature)

    def retrieve_parameter_rule(
        self, signature: ParameterCompilationSign
    ) -> ParameterCompilationFunc:
        return self._parameters_registry.retrieve_rule(signature)

    def retrieve_initializer_rule(
        self, signature: InitializerCompilationSign
    ) -> InitializerCompilationFunc:
        return self._initializers_registry.retrieve_rule(signature)

    def compile(self, sc: Circuit) -> CompiledCircuit:
        if self.is_compiled(sc):
            return self.get_compiled_circuit(sc)
        return self.compile_pipeline(sc)

    @abstractmethod
    def compile_pipeline(self, sc: Circuit) -> CompiledCircuit:
        ...

    @abstractmethod
    def save(
        self,
        sym_filepath: Union[IO, os.PathLike, str],
        compiled_filepath: Union[IO, os.PathLike, str],
    ):
        ...

    @staticmethod
    @abstractmethod
    def load(
        sym_filepath: Union[IO, os.PathLike, str], tens_filepath: Union[IO, os.PathLike, str]
    ) -> "AbstractCompiler":
        ...
