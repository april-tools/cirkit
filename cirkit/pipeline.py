from typing import Any, Dict

from cirkit.backend.base import SUPPORTED_BACKENDS, LayerCompilationFunction, AbstractCompiler
from cirkit.symbolic.functional import integrate, multiply, differentiate
from cirkit.symbolic.registry import SymbLayerOperatorFunction, SymbOperatorRegistry
from cirkit.symbolic.symb_circuit import SymbCircuit
from cirkit.symbolic.symb_layers import AbstractSymbLayerOperator


class PipelineContext:
    def __init__(self, backend: str = 'torch', **backend_kwargs):
        if backend not in SUPPORTED_BACKENDS:
            raise NotImplementedError(f"Backend '{backend}' is not implemented")
        self._backend = backend
        self._backend_kwargs = backend_kwargs
        self._symb_registry = SymbOperatorRegistry()
        self._compiler = retrieve_compiler(backend, **backend_kwargs)
        self._tensorized_to_symb: Dict[Any, SymbCircuit] = {}

    def register_operator_rule(self, op: AbstractSymbLayerOperator, func: SymbLayerOperatorFunction):
        self._symb_registry.register_rule(op, func)

    def register_compilation_rule(self, func: LayerCompilationFunction):
        self._compiler.register_rule(func)

    def compile(self, sc: SymbCircuit) -> Any:
        tc = self._compiler.compile(sc, )
        self._tensorized_to_symb[tc] = sc
        return tc

    def integrate(self, tc: Any) -> Any:
        sc = self._tensorized_to_symb[tc]
        integral_sc = integrate(sc, self._symb_registry)
        return self.compile(integral_sc)

    def multiply(self, lhs_tc: Any, rhs_tc: Any) -> Any:
        lhs_sc = self._tensorized_to_symb[lhs_tc]
        rhs_sc = self._tensorized_to_symb[rhs_tc]
        product_sc = multiply(lhs_sc, rhs_sc, self._symb_registry)
        return self.compile(product_sc)

    def differentiate(self, tc: Any) -> Any:
        sc = self._tensorized_to_symb[tc]
        differential_sc = differentiate(sc, self._symb_registry)
        return self.compile(differential_sc)


def retrieve_compiler(backend: str, **backend_kwargs) -> AbstractCompiler:
    if backend not in SUPPORTED_BACKENDS:
        raise NotImplementedError(f"Backend '{backend}' is not implemented")
    if backend == 'torch':
        from cirkit.backend.torch.compiler import Compiler
        return Compiler(**backend_kwargs)
    assert False
