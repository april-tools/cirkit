from typing import Any

from cirkit.backend.base import SUPPORTED_BACKENDS, LayerCompilationFunction, AbstractCompiler
from cirkit.symbolic.functional import integrate, multiply, differentiate
from cirkit.symbolic.registry import SymbLayerOperatorFunction, SymbOperatorRegistry
from cirkit.symbolic.symb_circuit import SymbCircuit
from cirkit.symbolic.symb_layers import AbstractSymbLayerOperator


class PipelineContext:
    def __init__(self, backend: str = 'torch'):
        if backend not in SUPPORTED_BACKENDS:
            raise NotImplementedError(f"Backend '{backend}' is not implemented")
        self._backend = backend
        self._symb_registry = SymbOperatorRegistry()
        self._compiler = retrieve_compiler(backend)

    def register_operator_rule(self, op: AbstractSymbLayerOperator, func: SymbLayerOperatorFunction):
        self._symb_registry.register_rule(op, func)

    def register_compilation_rule(self, func: LayerCompilationFunction):
        self._compiler.register_rule(func)

    def compile(self, sc: SymbCircuit, **opt_kwargs) -> Any:
        return self._compiler.compile(sc, **opt_kwargs)

    def integrate(self, sc: SymbCircuit) -> SymbCircuit:
        return integrate(sc, self._symb_registry)

    def multiply(self, lhs_sc: SymbCircuit, rhs_sc) -> SymbCircuit:
        return multiply(lhs_sc, rhs_sc, self._symb_registry)

    def differentiate(self, sc: SymbCircuit) -> SymbCircuit:
        return differentiate(sc, self._symb_registry)


def retrieve_compiler(backend: str) -> AbstractCompiler:
    if backend not in SUPPORTED_BACKENDS:
        raise NotImplementedError(f"Backend '{backend}' is not implemented")
    if backend == 'torch':
        from cirkit.backend.torch.compiler import Compiler
        return Compiler()
    assert False
