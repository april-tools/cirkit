import os
from abc import ABC, abstractmethod
from typing import IO, Any, Callable, Dict, Optional, Protocol, Type, Union

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import Layer
from cirkit.symbolic.params import AbstractParameter

LayerCompilationSign = Type[Layer]


class LayerCompilationFunc(Protocol):
    def __call__(self, compiler: "AbstractCompiler", sl: Layer, **kwargs) -> Any:
        ...


ParameterCompilationSign = Type[AbstractParameter]


class ParameterCompilationFunc(Protocol):
    def __call__(self, compiler: "AbstractCompiler", p: AbstractParameter, **kwargs) -> Any:
        ...


class CompilationRuleNotFound(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


SUPPORTED_BACKENDS = ["torch"]


class CompilationContext:
    def __init__(self):
        self._map: Dict[Circuit, Any] = {}
        self._inv_map: Dict[Any, Circuit] = {}

    def is_compiled(self, sc: Circuit) -> bool:
        return sc in self._map

    def has_compiled(self, tc: Any) -> bool:
        return tc in self._inv_map

    def get_compiled_circuit(self, sc: Circuit) -> Any:
        return self._map[sc]

    def get_symbolic_circuit(self, tc: Any) -> Circuit:
        return self._inv_map[tc]

    def register_compiled_circuit(self, sc: Circuit, tc: Any):
        self._map[sc] = tc
        self._inv_map[tc] = sc


class CompilerRegistry:
    def __init__(
        self,
        layer_rules: Optional[Dict[LayerCompilationSign, LayerCompilationFunc]] = None,
        parameter_rules: Optional[Dict[ParameterCompilationSign, ParameterCompilationFunc]] = None,
    ):
        self._layer_rules = {} if layer_rules is None else layer_rules
        self._parameter_rules = {} if parameter_rules is None else parameter_rules

    @staticmethod
    def _is_signature_valid(func: Callable) -> bool:
        args = func.__annotations__
        if "return" not in args or "compiler" not in args or len(args) != 3:
            return False
        if not issubclass(args["compiler"], AbstractCompiler):
            return False
        return True

    def add_layer_rule(self, func: LayerCompilationFunc):
        if not self._is_signature_valid(func):
            raise ValueError("The function is not a symbolic layer compilation rule")
        args = func.__annotations__
        arg_names = list(filter(lambda a: a not in ("return", "compiler"), args.keys()))
        layer_cls = args[arg_names[0]]
        if not issubclass(layer_cls, Layer):
            raise ValueError("The function is not a symbolic layer compilation rule")
        self._layer_rules[layer_cls] = func

    def add_parameter_rule(self, func: ParameterCompilationFunc):
        if not self._is_signature_valid(func):
            raise ValueError("The function is not a symbolic parameter compilation rule")
        args = func.__annotations__
        arg_names = list(filter(lambda a: a not in ("return", "compiler"), args.keys()))
        param_cls = args[arg_names[0]]
        if not issubclass(param_cls, AbstractParameter):
            raise ValueError("The function is not a symbolic parameter compilation rule")
        self._parameter_rules[param_cls] = func

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


class AbstractCompiler(ABC):
    def __init__(self, registry: CompilerRegistry, **flags):
        self._registry = registry
        self._flags = flags
        self._context = CompilationContext()

    def is_compiled(self, sc: Circuit) -> bool:
        return self._context.is_compiled(sc)

    def has_symbolic(self, tc: Any) -> bool:
        return self._context.has_compiled(tc)

    def get_compiled_circuit(self, sc: Circuit) -> Any:
        return self._context.get_compiled_circuit(sc)

    def get_symbolic_circuit(self, tc: Any) -> Circuit:
        return self._context.get_symbolic_circuit(tc)

    def register_compiled_circuit(self, sc: Circuit, tc: Any):
        self._context.register_compiled_circuit(sc, tc)

    def add_layer_rule(self, func: LayerCompilationFunc):
        self._registry.add_layer_rule(func)

    def add_parameter_rule(self, func: ParameterCompilationFunc):
        self._registry.add_parameter_rule(func)

    def retrieve_layer_rule(self, signature: LayerCompilationSign) -> LayerCompilationFunc:
        return self._registry.retrieve_layer_rule(signature)

    def retrieve_parameter_rule(
        self, signature: ParameterCompilationSign
    ) -> ParameterCompilationFunc:
        return self._registry.retrieve_parameter_rule(signature)

    def compile(self, sc: Circuit) -> Any:
        if self.is_compiled(sc):
            return self.get_compiled_circuit(sc)
        tc = self.compile_pipeline(sc)
        self._context.register_compiled_circuit(sc, tc)
        return tc

    @abstractmethod
    def compile_pipeline(self, sc: Circuit) -> Any:
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
