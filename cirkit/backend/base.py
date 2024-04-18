import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, Union, IO

from cirkit.symbolic.sym_circuit import SymCircuit
from cirkit.symbolic.sym_layers import SymLayer
from cirkit.symbolic.sym_params import SymParameter, AbstractSymParameter

LayerCompilationSignature = Union[Type[SymLayer]]
LayerCompilationFunction = Callable[[SymLayer, 'AbstractCompiler'], Any]


SUPPORTED_BACKENDS = ['torch']


class CompilationContext:
    def __init__(self):
        self._map: Dict[SymCircuit, Any] = {}
        self._inv_map: Dict[Any, SymCircuit] = {}

    def is_compiled(self, sc: SymCircuit) -> bool:
        return sc in self._map

    def has_compiled(self, tc: Any) -> bool:
        return tc in self._inv_map

    def get_compiled_circuit(self, sc: SymCircuit) -> Any:
        return self._map[sc]

    def get_symbolic_circuit(self, tc: Any) -> SymCircuit:
        return self._inv_map[tc]

    def register_compiled_circuit(self, sc: SymCircuit, tc: Any):
        self._map[sc] = tc
        self._inv_map[tc] = sc


class CompilerRegistry:
    def __init__(
        self,
        default_rules: Optional[Dict[LayerCompilationSignature, LayerCompilationFunction]] = None,
    ):
        self._rules = {} if default_rules is None else default_rules

    def register_rule(self, func: LayerCompilationFunction):
        args = func.__annotations__
        if "return" not in args or "compiler" not in args or len(args) != 3:
            raise ValueError("The function is not a symbolic layer compilation rule")
        if not issubclass(args["compiler"], AbstractCompiler):
            raise ValueError("The function is not a symbolic layer compilation rule")
        arg_names = list(filter(lambda a: a not in ("return", "compiler"), args.keys()))
        sym_cls = args[arg_names[0]]
        if not issubclass(sym_cls, (SymLayer, SymParameter)):
            raise ValueError("The function is not a symbolic layer compilation rule")
        self._rules[sym_cls] = func


class AbstractCompiler(ABC):
    def __init__(self, registry: CompilerRegistry, **flags):
        self._registry = registry
        self._flags = flags
        self._context = CompilationContext()

    def is_compiled(self, sc: SymCircuit) -> bool:
        return self._context.is_compiled(sc)

    def has_symbolic(self, tc: Any) -> bool:
        return self._context.has_compiled(tc)

    def get_compiled_circuit(self, sc: SymCircuit) -> Any:
        return self._context.get_compiled_circuit(sc)

    def get_symbolic_circuit(self, tc: Any) -> SymCircuit:
        return self._context.get_symbolic_circuit(tc)

    def register_compiled_circuit(self, sc: SymCircuit, tc: Any):
        self._context.register_compiled_circuit(sc, tc)

    def register_rule(self, func: LayerCompilationFunction):
        self._registry.register_rule(func)

    def compile(self, sc: SymCircuit) -> Any:
        if self.is_compiled(sc):
            return self.get_compiled_circuit(sc)
        tc = self.compile_pipeline(sc)
        self._context.register_compiled_circuit(sc, tc)
        return tc

    @abstractmethod
    def compile_pipeline(self, sc: SymCircuit) -> Any:
        ...

    @abstractmethod
    def compile_learnable_parameter(self, sym_param: AbstractSymParameter) -> Any:
        ...

    @abstractmethod
    def save(self, sym_filepath: Union[IO, os.PathLike, str], compiled_filepath: Union[IO, os.PathLike, str]):
        ...

    @staticmethod
    @abstractmethod
    def load(sym_filepath: Union[IO, os.PathLike, str], tens_filepath: Union[IO, os.PathLike, str]) -> 'AbstractCompiler':
        ...
