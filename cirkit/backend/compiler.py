from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar, cast

from cirkit.backend.registry import CompilerRegistry
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import Initializer
from cirkit.symbolic.layers import Layer
from cirkit.symbolic.parameters import ParameterNode
from cirkit.utils.algorithms import BiMap

SUPPORTED_BACKENDS = ["torch"]

CompiledCircuit = TypeVar("CompiledCircuit")
GateFunction = TypeVar("GateFunction")


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


LayerCompilationSign = type[Layer]
ParameterCompilationSign = type[ParameterNode]
InitializerCompilationSign = type[Initializer]


class LayerCompilationFunc(Protocol):
    """The layer compilation function protocol."""

    def __call__(self, compiler: "AbstractCompiler", sl: Layer, **kwargs) -> Any:
        """Compile a symbolic layer, given a compiler.

        Args:
            compiler: The compiler.
            sl: The symbolic layer.
            **kwargs: The optional arguments for the compilation.

        Returns:
            A representation of the compiled layer, which depends on the chosen compilation backend.
        """


class ParameterCompilationFunc(Protocol):
    """The parameter node compilation function protocol."""

    def __call__(self, compiler: "AbstractCompiler", p: ParameterNode, **kwargs) -> Any:
        """Compile a symbolic parameter node, given a compiler.

        Args:
            compiler: The compiler.
            p: The symbolic parameter node.
            **kwargs: The optional arguments for the compilation.

        Returns:
            A representation of the compiled parameter node,
                which depends on the chosen compilation backend.
        """


class InitializerCompilationFunc(Protocol):
    """The initialization method compilation function protocol."""

    def __call__(self, compiler: "AbstractCompiler", init: Initializer, **kwargs) -> Any:
        """Compile a symbolic initializer, given a compiler.

        Args:
            compiler: The compiler.
            init: The symbolic initializer.
            **kwargs: The optional arguments for the compilation.

        Returns:
            A representation of the compiled initializer,
                which depends on the chosen compilation backend.
        """


class CompilationRuleNotFound(Exception):
    """An exception that is raised when a compilation rule is not found."""

    def __init__(self, msg: str):
        """Initializes a compilation rule not found exception.

        Args:
            msg: The message of the exception.
        """
        super().__init__(msg)


class CompilerLayerRegistry(CompilerRegistry[LayerCompilationSign, LayerCompilationFunc]):
    @classmethod
    def _validate_rule_function(cls, func: LayerCompilationFunc) -> bool:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return issubclass(ann[args[-1]], Layer)

    @classmethod
    def _retrieve_signature(cls, func: LayerCompilationFunc) -> LayerCompilationSign:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return cast(LayerCompilationSign, ann[args[-1]])


class CompilerParameterRegistry(
    CompilerRegistry[ParameterCompilationSign, ParameterCompilationFunc]
):
    @classmethod
    def _validate_rule_function(cls, func: ParameterCompilationFunc) -> bool:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return issubclass(ann[args[-1]], ParameterNode)

    @classmethod
    def _retrieve_signature(cls, func: ParameterCompilationFunc) -> ParameterCompilationSign:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return cast(ParameterCompilationSign, ann[args[-1]])


class CompilerInitializerRegistry(
    CompilerRegistry[InitializerCompilationSign, InitializerCompilationFunc]
):
    @classmethod
    def _validate_rule_function(cls, func: InitializerCompilationFunc) -> bool:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return issubclass(ann[args[-1]], Initializer)

    @classmethod
    def _retrieve_signature(cls, func: ParameterCompilationFunc) -> InitializerCompilationSign:
        ann = func.__annotations__.copy()
        del ann["return"]
        args = tuple(ann.keys())
        return cast(InitializerCompilationSign, ann[args[-1]])


CompilerGateFunctionRegistry = dict[str, GateFunction]


class AbstractCompiler(ABC):
    def __init__(
        self,
        layers_registry: CompilerLayerRegistry,
        parameters_registry: CompilerParameterRegistry,
        initializers_registry: CompilerInitializerRegistry,
        gate_function_registry: CompilerGateFunctionRegistry | None = None,
        **flags,
    ):
        self._layers_registry = layers_registry
        self._parameters_registry = parameters_registry
        self._initializers_registry = initializers_registry
        self._gate_function_registry = (
            {} if gate_function_registry is None else gate_function_registry
        )
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

    def add_gate_function(self, function_id: str, gate_function: GateFunction):
        self._gate_function_registry[function_id] = gate_function

    def get_gate_function(self, function_id: str) -> GateFunction:
        return self._gate_function_registry[function_id]

    def compile(self, sc: Circuit) -> CompiledCircuit:
        if self.is_compiled(sc):
            return self.get_compiled_circuit(sc)
        return self.compile_pipeline(sc)

    @abstractmethod
    def compile_pipeline(self, sc: Circuit) -> CompiledCircuit:
        ...
