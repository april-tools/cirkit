from contextlib import AbstractContextManager
from contextvars import Token
from types import TracebackType
from typing import Any, Dict, Iterable, Optional, Type

from cirkit.backend.base import _SUPPORTED_BACKENDS, AbstractCompiler, LayerCompilationFunction
from cirkit.symbolic.functional import _SYM_OPERATOR_REGISTRY, differentiate, integrate, multiply
from cirkit.symbolic.registry import SymLayerOperatorFunction, SymOperatorRegistry
from cirkit.symbolic.sym_circuit import SymCircuit
from cirkit.symbolic.sym_layers import AbstractSymLayerOperator


class CompiledCircuitMap:
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


class PipelineContext(AbstractContextManager):
    def __init__(self, backend: str = "torch", **backend_kwargs):
        if backend not in _SUPPORTED_BACKENDS:
            raise NotImplementedError(f"Backend '{backend}' is not implemented")
        # Backend specs
        self._backend = backend
        self._backend_kwargs = backend_kwargs

        # Symbolic operator registry (and token for context management)
        self._symb_registry = SymOperatorRegistry()
        self._symb_registry_token: Optional[Token[SymOperatorRegistry]] = None

        # Compiler objects and data structures
        self._compiled_map = CompiledCircuitMap()
        self._compiler = retrieve_compiler(backend, **backend_kwargs)

    def __getitem__(self, sc: SymCircuit) -> Any:
        return self._compiled_map.get_compiled_circuit(sc)

    def __enter__(self) -> "PipelineContext":
        self._symb_registry_token = _SYM_OPERATOR_REGISTRY.set(self._symb_registry)
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        _SYM_OPERATOR_REGISTRY.reset(self._symb_registry_token)
        self._symb_registry_token = None
        return None

    def register_operator_rule(self, op: AbstractSymLayerOperator, func: SymLayerOperatorFunction):
        self._symb_registry.register_rule(op, func)

    def register_compilation_rule(self, func: LayerCompilationFunction):
        self._compiler.register_rule(func)

    def compile(self, sc: SymCircuit) -> Any:
        if self._compiled_map.is_compiled(sc):
            return self._compiled_map.get_compiled_circuit(sc)
        tc = self._compiler.compile(sc)
        self._compiled_map.register_compiled_circuit(sc, tc)
        return tc

    def integrate(self, tc: Any, scope: Optional[Iterable[int]] = None) -> Any:
        if not self._compiled_map.has_compiled(tc):
            raise ValueError("The given compiled circuit is not known in this pipeline")
        sc = self._compiled_map.get_symbolic_circuit(tc)
        integral_sc = integrate(sc, scope=scope, registry=self._symb_registry)
        return self.compile(integral_sc)

    def multiply(self, lhs_tc: Any, rhs_tc: Any) -> Any:
        if not self._compiled_map.has_compiled(lhs_tc):
            raise ValueError("The given LHS compiled circuit is not known in this pipeline")
        if not self._compiled_map.has_compiled(rhs_tc):
            raise ValueError("The given RHS compiled circuit is not known in this pipeline")
        lhs_sc = self._compiled_map.get_symbolic_circuit(lhs_tc)
        rhs_sc = self._compiled_map.get_symbolic_circuit(rhs_tc)
        product_sc = multiply(lhs_sc, rhs_sc, registry=self._symb_registry)
        return self.compile(product_sc)

    def differentiate(self, tc: Any) -> Any:
        if not self._compiled_map.has_compiled(tc):
            raise ValueError("The given compiled circuit is not known in this pipeline")
        sc = self._compiled_map.get_symbolic_circuit(tc)
        differential_sc = differentiate(sc, registry=self._symb_registry)
        return self.compile(differential_sc)


def retrieve_compiler(backend: str, **backend_kwargs) -> AbstractCompiler:
    if backend not in _SUPPORTED_BACKENDS:
        raise NotImplementedError(f"Backend '{backend}' is not implemented")
    if backend == "torch":
        from cirkit.backend.torch.compiler import TorchCompiler

        return TorchCompiler(**backend_kwargs)
    assert False
