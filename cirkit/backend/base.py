from abc import ABC, abstractmethod
from typing import Union, Callable, Any, Optional, Dict, Type

from cirkit.symbolic.symb_circuit import SymbCircuit
from cirkit.symbolic.symb_layers import SymbLayer
from cirkit.symbolic.symb_params import SymbParameter

LayerCompilationSignature = Union[Type[SymbLayer], Type[SymbParameter]]
LayerCompilationFunction = Callable[[Union[SymbLayer, SymbParameter]], Any]


SUPPORTED_BACKENDS = ['torch']


class CompilationRegistry:
    def __init__(
            self,
            default_rules: Optional[Dict[LayerCompilationSignature, LayerCompilationFunction]] = None
    ):
        self._rules = {} if default_rules is None else default_rules

    def register_rule(self, func: LayerCompilationFunction):
        args = func.__annotations__
        if 'return' not in args or len(args) != 2:
            raise ValueError("The function is not a symbolic layer compilation rule")
        arg_names = list(filter(lambda a: a != 'return', args.keys()))
        symb_cls = args[arg_names[0]]
        if not issubclass(symb_cls, (SymbLayer, SymbParameter)):
            raise ValueError("The function is not a symbolic layer compilation rule")
        self._rules[symb_cls] = func


class AbstractCompiler(ABC):
    def __init__(self, registry: CompilationRegistry):
        self._registry = registry

    def register_rule(self, func: LayerCompilationFunction):
        self._registry.register_rule(func)

    @abstractmethod
    def compile(self, symb_circuit: SymbCircuit, **opt_kwargs):
        ...
