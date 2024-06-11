import os
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from types import TracebackType
from typing import IO, Any, Iterable, Optional, Type, Union

from cirkit.backend.base import SUPPORTED_BACKENDS, AbstractCompiler, LayerCompilationFunc, \
    ParameterCompilationFunc
import cirkit.symbolic.functional as SF
from cirkit.symbolic.registry import OperatorRegistry
from cirkit.symbolic.operators import LayerOperatorFunc
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import AbstractLayerOperator


class PipelineContext(AbstractContextManager):
    def __init__(self, backend: str = "torch", **backend_kwargs):
        if backend not in SUPPORTED_BACKENDS:
            raise NotImplementedError(f"Backend '{backend}' is not implemented")
        # Backend specs
        self._backend = backend
        self._backend_kwargs = backend_kwargs

        # Symbolic operator registry
        self._op_registry = OperatorRegistry.from_default_rules()

        # Get the compiler, which is backend-dependent
        self._compiler = retrieve_compiler(backend, **backend_kwargs)

        # The token used to restore the pipeline context
        self._token: Optional[Token[PipelineContext]] = None

    @classmethod
    def from_default_backend(cls) -> 'PipelineContext':
        return PipelineContext(backend="torch", fold=True, einsum=True)

    def __getitem__(self, sc: Circuit) -> Any:
        return self._compiler.get_compiled_circuit(sc)

    def __enter__(self) -> "PipelineContext":
        self._op_registry.__enter__()
        self._token = _PIPELINE_CONTEXT.set(self)
        return self

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        ret = self._op_registry.__exit__(__exc_type, __exc_value, __traceback)
        _PIPELINE_CONTEXT.reset(self._token)
        self._token = None
        return ret

    def save(
        self,
        sym_fp: Union[IO, os.PathLike, str],
        state_fp: Optional[Union[IO, os.PathLike, str]] = None,
    ):
        ...

    @staticmethod
    def load(
        sym_fp: Union[IO, os.PathLike, str], state_fp: Optional[Union[IO, os.PathLike, str]] = None
    ) -> "PipelineContext":
        ...

    def add_operator_rule(self, op: AbstractLayerOperator, func: LayerOperatorFunc):
        self._op_registry.add_rule(op, func)

    def add_layer_compilation_rule(self, func: LayerCompilationFunc):
        self._compiler.add_layer_rule(func)

    def add_parameter_compilation_rule(self, func: ParameterCompilationFunc):
        self._compiler.add_parameter_rule(func)

    def compile(self, sc: Circuit) -> Any:
        return self._compiler.compile(sc)

    def integrate(self, cc: Any, scope: Optional[Iterable[int]] = None) -> Any:
        if not self._compiler.has_symbolic(cc):
            raise ValueError("The given compiled circuit is not known in this pipeline")
        sc = self._compiler.get_symbolic_circuit(cc)
        int_sc = SF.integrate(sc, scope=scope, registry=self._op_registry)
        return self.compile(int_sc)

    def multiply(self, lhs_cc: Any, rhs_cc: Any) -> Any:
        if not self._compiler.has_symbolic(lhs_cc):
            raise ValueError("The given LHS compiled circuit is not known in this pipeline")
        if not self._compiler.has_symbolic(rhs_cc):
            raise ValueError("The given RHS compiled circuit is not known in this pipeline")
        lhs_sc = self._compiler.get_symbolic_circuit(lhs_cc)
        rhs_sc = self._compiler.get_symbolic_circuit(rhs_cc)
        prod_sc = SF.multiply(lhs_sc, rhs_sc, registry=self._op_registry)
        return self.compile(prod_sc)

    def differentiate(self, cc: Any) -> Any:
        if not self._compiler.has_symbolic(cc):
            raise ValueError("The given compiled circuit is not known in this pipeline")
        sc = self._compiler.get_symbolic_circuit(cc)
        diff_sc = SF.differentiate(sc, registry=self._op_registry)
        return self.compile(diff_sc)


def compile(sc: Circuit, ctx: Optional[PipelineContext] = None) -> Any:
    if ctx is None:
        ctx = _PIPELINE_CONTEXT.get()
    return ctx.compile(sc)


def integrate(cc: Any, scope: Optional[Iterable[int]] = None, ctx: Optional[PipelineContext] = None) -> Any:
    if ctx is None:
        ctx = _PIPELINE_CONTEXT.get()
    return ctx.integrate(cc, scope=scope)


def multiply(lhs_cc: Any, rhs_cc: Any, ctx: Optional[PipelineContext] = None) -> Any:
    if ctx is None:
        ctx = _PIPELINE_CONTEXT.get()
    return ctx.multiply(lhs_cc, rhs_cc)


def differentiate(cc: Any, ctx: Optional[PipelineContext] = None) -> Any:
    if ctx is None:
        ctx = _PIPELINE_CONTEXT.get()
    return ctx.differentiate(cc)


def retrieve_compiler(backend: str, **backend_kwargs) -> AbstractCompiler:
    if backend not in SUPPORTED_BACKENDS:
        raise NotImplementedError(f"Backend '{backend}' is not implemented")
    if backend == "torch":
        from cirkit.backend.torch.compiler import TorchCompiler

        return TorchCompiler(**backend_kwargs)
    assert False


# Context variable holding the current global pipeline.
# This is updated when entering a pipeline context.
_PIPELINE_CONTEXT: ContextVar[PipelineContext] = ContextVar(
    "_PIPELINE_CONTEXT", default=PipelineContext.from_default_backend()
)
