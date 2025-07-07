import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import IO, Any, Protocol, TypeVar

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import Initializer
from cirkit.symbolic.layers import Layer
from cirkit.symbolic.parameters import ParameterNode
from cirkit.utils.algorithms import BiMap

CompiledCircuit = TypeVar("CompiledCircuit")
LayerCompilationSign = type[Layer]
ParameterCompilationSign = type[ParameterNode]
InitializerCompilationSign = type[Initializer]


class LayerCompilationFunc(Protocol):
    def __call__(self, compiler: "AbstractCompiler", sl: Layer, **kwargs) -> Any: ...


class ParameterCompilationFunc(Protocol):
    def __call__(self, compiler: "AbstractCompiler", p: ParameterNode, **kwargs) -> Any: ...


class InitializerCompilationFunc(Protocol):
    def __call__(self, compiler: "AbstractCompiler", init: Initializer, **kwargs) -> Any: ...


class CompilationRuleNotFound(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


SUPPORTED_BACKENDS = ["torch"]


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


class CompilerRegistry:
    def __init__(
        self,
        layer_rules: dict[LayerCompilationSign, LayerCompilationFunc] | None = None,
        parameter_rules: dict[ParameterCompilationSign, ParameterCompilationFunc] | None = None,
        initializer_rules: None | (
            dict[InitializerCompilationSign, InitializerCompilationFunc]
        ) = None,
    ):
        self._layer_rules = {} if layer_rules is None else layer_rules
        self._parameter_rules = {} if parameter_rules is None else parameter_rules
        self._initializer_rules = {} if initializer_rules is None else initializer_rules

    @staticmethod
    def _validate_rule_sign(func: Callable, sym_cls: type) -> type | None:
        args = func.__annotations__
        if "return" not in args or "compiler" not in args or len(args) != 3:
            return None
        if not issubclass(args["compiler"], AbstractCompiler):
            return None
        arg_names = list(filter(lambda a: a not in ("return", "compiler"), args.keys()))
        found_sym_cls = args[arg_names[0]]
        if not issubclass(found_sym_cls, sym_cls):
            return None
        return found_sym_cls

    def add_layer_rule(self, func: LayerCompilationFunc):
        layer_cls: type[Layer] | None = self._validate_rule_sign(func, Layer)
        if layer_cls is None:
            raise ValueError("The function is not a symbolic layer compilation rule")
        self._layer_rules[layer_cls] = func

    def add_parameter_rule(self, func: ParameterCompilationFunc):
        param_cls: type[ParameterNode] | None = self._validate_rule_sign(func, ParameterNode)
        if param_cls is None:
            raise ValueError("The function is not a symbolic parameter compilation rule")
        self._parameter_rules[param_cls] = func

    def add_initializer_rule(self, func: InitializerCompilationFunc):
        init_cls: type[Initializer] | None = self._validate_rule_sign(func, Initializer)
        if init_cls is None:
            raise ValueError("The function is not a symbolic initializer compilation rule")
        self._initializer_rules[init_cls] = func

    def retrieve_layer_rule(self, signature: LayerCompilationSign) -> LayerCompilationFunc:
        if signature not in self._layer_rules:
            raise CompilationRuleNotFound(
                f"Layer compilation rule for signature '{signature}' not found"
            )
        return self._layer_rules[signature]

    def retrieve_parameter_rule(
        self, signature: ParameterCompilationSign
    ) -> ParameterCompilationFunc:
        if signature not in self._parameter_rules:
            raise CompilationRuleNotFound(
                f"Parameter compilation rule for signature '{signature}' not found"
            )
        return self._parameter_rules[signature]

    def retrieve_initializer_rule(
        self, signature: InitializerCompilationSign
    ) -> InitializerCompilationFunc:
        if signature not in self._initializer_rules:
            raise CompilationRuleNotFound(
                f"Initializer compilation rule for signature '{signature}' not found"
            )
        return self._initializer_rules[signature]


class AbstractCompiler(ABC):
    def __init__(self, registry: CompilerRegistry, **flags):
        self._registry = registry
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
        self._registry.add_layer_rule(func)

    def add_parameter_rule(self, func: ParameterCompilationFunc):
        self._registry.add_parameter_rule(func)

    def add_initializer_rule(self, func: InitializerCompilationFunc):
        self._registry.add_initializer_rule(func)

    def retrieve_layer_rule(self, signature: LayerCompilationSign) -> LayerCompilationFunc:
        return self._registry.retrieve_layer_rule(signature)

    def retrieve_parameter_rule(
        self, signature: ParameterCompilationSign
    ) -> ParameterCompilationFunc:
        return self._registry.retrieve_parameter_rule(signature)

    def retrieve_initializer_rule(
        self, signature: InitializerCompilationSign
    ) -> InitializerCompilationFunc:
        return self._registry.retrieve_initializer_rule(signature)

    def compile(self, sc: Circuit) -> CompiledCircuit:
        if self.is_compiled(sc):
            return self.get_compiled_circuit(sc)
        return self.compile_pipeline(sc)

    @abstractmethod
    def compile_layer(self, sl: Layer) -> Any: ...

    @abstractmethod
    def compile_pipeline(self, sc: Circuit) -> CompiledCircuit: ...

    @abstractmethod
    def save(
        self,
        sym_filepath: IO | os.PathLike | str,
        compiled_filepath: IO | os.PathLike | str,
    ): ...

    @staticmethod
    @abstractmethod
    def load(
        sym_filepath: IO | os.PathLike | str, tens_filepath: IO | os.PathLike | str
    ) -> "AbstractCompiler": ...
