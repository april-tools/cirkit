from typing import Dict, Type, Tuple, Callable, Optional, Any

from cirkit.backend.base import SUPPORTED_BACKENDS, CompilationFunction, AbstractCompiler
from cirkit.symbolic.symb_circuit import SymbCircuit
from cirkit.symbolic.symb_layers import SymbLayer, SymbLayerOperator

SymbOperatorSignature = Tuple[Type[SymbLayer], ...]
SymbOperatorFunction = Callable[[SymbLayer, ...], SymbLayer]


class SymbOperatorRegistry:
    def __init__(
            self,
            default_operators: Optional[Dict[SymbOperatorSignature, SymbOperatorFunction]] = None,
            commutative: bool = False
    ):
        self._operators = {} if default_operators is None else default_operators
        self._commutative = commutative

    def register_operator(self, func: SymbOperatorFunction):
        args = func.__annotations__
        arg_names = list(filter(lambda a: a != 'return', args.keys()))
        if len(arg_names) == 0 or not all(issubclass(args[a], SymbLayer) for a in arg_names):
            raise ValueError("The function is not an operator over symbolic layers")
        if len(arg_names) == 2:  # binary operator (special case as to deal with commutative operators)
            lhs_symb_cls = args[arg_names[0]]
            rhs_symb_cls = args[arg_names[1]]
            self._operators[(lhs_symb_cls, rhs_symb_cls)] = func
            if self._commutative and lhs_symb_cls != rhs_symb_cls:
                self._operators[(rhs_symb_cls, lhs_symb_cls)] = lambda rhs, lhs: func(lhs, rhs)
        else:  # n-ary operator
            symb_signature = tuple(args[a] for a in arg_names)
            self._operators[symb_signature] = func


_DEFAULT_SYMB_OPERATOR_REGISTRY = {
    SymbLayerOperator.INTEGRATION: SymbOperatorRegistry(
        default_operators={
        }
    ),
    SymbLayerOperator.DIFFERENTIATION: SymbOperatorRegistry(
        default_operators={
        }
    ),
    SymbLayerOperator.KRONECKER: SymbOperatorRegistry(
        default_operators={
        }
    )
}


class PipelineContext:
    def __init__(self, backend: str = 'torch'):
        if backend not in SUPPORTED_BACKENDS:
            raise NotImplementedError(f"Backend '{backend}' is not implemented")
        self._backend = backend
        self._symb_operators = _DEFAULT_SYMB_OPERATOR_REGISTRY
        self._compiler = retrieve_compiler(backend)

    def register_operator_rule(self, op: SymbLayerOperator, func: SymbOperatorFunction):
        self._symb_operators[op].register_operator(func)

    def register_compilation_rule(self, func: CompilationFunction):
        self._compiler.register_rule(func)

    def compile(self, symb_circuit: SymbCircuit, **opt_kwargs) -> Any:
        return self._compiler.compile(symb_circuit, **opt_kwargs)


def retrieve_compiler(backend: str) -> AbstractCompiler:
    if backend not in SUPPORTED_BACKENDS:
        raise NotImplementedError(f"Backend '{backend}' is not implemented")
    if backend == 'torch':
        from cirkit.backend.torch.compiler import Compiler
        return Compiler()
    assert False
