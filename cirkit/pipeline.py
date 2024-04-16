from contextlib import AbstractContextManager
from contextvars import Token, ContextVar
from types import TracebackType
from typing import Any, Iterable, Optional, Type

from cirkit.backend.base import SUPPORTED_BACKENDS, AbstractCompiler, LayerCompilationFunction
from cirkit.symbolic.functional import differentiate, integrate, multiply
from cirkit.symbolic.registry import SymLayerOperatorFunction, SymOperatorRegistry
from cirkit.symbolic.sym_circuit import SymCircuit
from cirkit.symbolic.sym_layers import AbstractSymLayerOperator


# Context variable containing the symbolic operator registry.
# This is updated when entering a pipeline context.
_SYM_OPERATOR_REGISTRY: ContextVar[SymOperatorRegistry] = ContextVar(
    "_SYM_OPERATOR_REGISTRY", default=SymOperatorRegistry()
)


class PipelineContext(AbstractContextManager):
    def __init__(self, backend: str = "torch", **backend_kwargs):
        if backend not in SUPPORTED_BACKENDS:
            raise NotImplementedError(f"Backend '{backend}' is not implemented")
        # Backend specs
        self._backend = backend
        self._backend_kwargs = backend_kwargs

        # Symbolic operator registry (and token for context management)
        self._sym_registry = SymOperatorRegistry()
        self._sym_registry_token: Optional[Token[SymOperatorRegistry]] = None

        # Get the compiler, which is backend-dependent
        self._compiler = retrieve_compiler(backend, **backend_kwargs)

    def __getitem__(self, sc: SymCircuit) -> Any:
        return self._compiler.get_compiled_circuit(sc)

    def __enter__(self) -> "PipelineContext":
        self._sym_registry_token = _SYM_OPERATOR_REGISTRY.set(self._sym_registry)
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        _SYM_OPERATOR_REGISTRY.reset(self._sym_registry_token)
        self._sym_registry_token = None
        return None

    def register_operator_rule(self, op: AbstractSymLayerOperator, func: SymLayerOperatorFunction):
        self._sym_registry.register_rule(op, func)

    def register_compilation_rule(self, func: LayerCompilationFunction):
        self._compiler.register_rule(func)

    def compile(self, sc: SymCircuit) -> Any:
        return self._compiler.compile(sc)

    def integrate(self, tc: Any, scope: Optional[Iterable[int]] = None) -> Any:
        if not self._compiler.has_symbolic(tc):
            raise ValueError("The given compiled circuit is not known in this pipeline")
        sc = self._compiler.get_symbolic_circuit(tc)
        int_sc = integrate(sc, scope=scope, registry=self._sym_registry)
        return self.compile(int_sc)

    def multiply(self, lhs_tc: Any, rhs_tc: Any) -> Any:
        if not self._compiler.has_symbolic(lhs_tc):
            raise ValueError("The given LHS compiled circuit is not known in this pipeline")
        if not self._compiler.has_symbolic(rhs_tc):
            raise ValueError("The given RHS compiled circuit is not known in this pipeline")
        lhs_sc = self._compiler.get_symbolic_circuit(lhs_tc)
        rhs_sc = self._compiler.get_symbolic_circuit(rhs_tc)
        prod_sc = multiply(lhs_sc, rhs_sc, registry=self._sym_registry)
        return self.compile(prod_sc)

    def differentiate(self, tc: Any) -> Any:
        if not self._compiler.has_symbolic(tc):
            raise ValueError("The given compiled circuit is not known in this pipeline")
        sc = self._compiler.get_symbolic_circuit(tc)
        diff_sc = differentiate(sc, registry=self._sym_registry)
        return self.compile(diff_sc)


def retrieve_compiler(backend: str, **backend_kwargs) -> AbstractCompiler:
    if backend not in SUPPORTED_BACKENDS:
        raise NotImplementedError(f"Backend '{backend}' is not implemented")
    if backend == "torch":
        from cirkit.backend.torch.compiler import TorchCompiler

        return TorchCompiler(**backend_kwargs)
    assert False
