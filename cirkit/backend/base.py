from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, Union

from cirkit.symbolic.sym_circuit import SymCircuit
from cirkit.symbolic.sym_layers import SymLayer
from cirkit.symbolic.sym_params import SymParameter

LayerCompilationSignature = Union[Type[SymLayer]]
LayerCompilationFunction = Callable[[SymLayer, "AbstractCompiler"], Any]


_SUPPORTED_BACKENDS = ["torch"]


class CompilationRegistry:
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
        symb_cls = args[arg_names[0]]
        if not issubclass(symb_cls, (SymLayer, SymParameter)):
            raise ValueError("The function is not a symbolic layer compilation rule")
        self._rules[symb_cls] = func


class AbstractCompiler(ABC):
    def __init__(self, registry: CompilationRegistry, **flags):
        self._registry = registry
        self._flags = flags

    def register_rule(self, func: LayerCompilationFunction):
        self._registry.register_rule(func)

    @abstractmethod
    def compile(self, symb_circuit: SymCircuit):
        ...
