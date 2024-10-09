from collections.abc import Iterable
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from types import TracebackType

import cirkit.symbolic.functional as SF
from cirkit.backend.compiler import (
    SUPPORTED_BACKENDS,
    AbstractCompiler,
    CompiledCircuit,
    InitializerCompilationFunc,
    LayerCompilationFunc,
    ParameterCompilationFunc,
)
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import LayerOperator
from cirkit.symbolic.operators import LayerOperatorFunc
from cirkit.symbolic.registry import OperatorRegistry


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
        self._token: Token[PipelineContext] | None = None

    @classmethod
    def from_default_backend(cls) -> "PipelineContext":
        return PipelineContext(backend="torch", semiring="lse-sum", fold=True, optimize=True)

    def __getitem__(self, sc: Circuit) -> CompiledCircuit:
        return self._compiler.get_compiled_circuit(sc)

    def __enter__(self) -> "PipelineContext":
        self._op_registry.__enter__()
        self._token = _PIPELINE_CONTEXT.set(self)
        return self

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        ret = self._op_registry.__exit__(__exc_type, __exc_value, __traceback)
        _PIPELINE_CONTEXT.reset(self._token)
        self._token = None
        return ret

    def add_operator_rule(self, op: LayerOperator, func: LayerOperatorFunc):
        self._op_registry.add_rule(op, func)

    def add_layer_compilation_rule(self, func: LayerCompilationFunc):
        self._compiler.add_layer_rule(func)

    def add_parameter_compilation_rule(self, func: ParameterCompilationFunc):
        self._compiler.add_parameter_rule(func)

    def add_initializer_compilation_rule(self, func: InitializerCompilationFunc):
        self._compiler.add_initializer_rule(func)

    def compile(self, sc: Circuit) -> CompiledCircuit:
        return self._compiler.compile(sc)

    def is_compiled(self, sc: Circuit) -> bool:
        return self._compiler.is_compiled(sc)

    def has_symbolic(self, cc: CompiledCircuit) -> bool:
        return self._compiler.has_symbolic(cc)

    def get_compiled_circuit(self, sc: Circuit) -> CompiledCircuit:
        return self._compiler.get_compiled_circuit(sc)

    def get_symbolic_circuit(self, cc: CompiledCircuit) -> Circuit:
        return self._compiler.get_symbolic_circuit(cc)

    def concatenate(self, *cc: CompiledCircuit) -> CompiledCircuit:
        for i, cci in enumerate(cc):
            if not self._compiler.has_symbolic(cci):
                raise ValueError(f"The {i}-th given compiled circuit is not known in this pipeline")
        sc = [self._compiler.get_symbolic_circuit(cci) for cci in cc]
        cat_sc = SF.concatenate(*sc, registry=self._op_registry)
        return self.compile(cat_sc)

    def integrate(self, cc: CompiledCircuit, scope: Iterable[int] | None = None) -> CompiledCircuit:
        if not self._compiler.has_symbolic(cc):
            raise ValueError("The given compiled circuit is not known in this pipeline")
        sc = self._compiler.get_symbolic_circuit(cc)
        int_sc = SF.integrate(sc, scope=scope, registry=self._op_registry)
        return self.compile(int_sc)

    def multiply(self, lhs_cc: CompiledCircuit, rhs_cc: CompiledCircuit) -> CompiledCircuit:
        if not self._compiler.has_symbolic(lhs_cc):
            raise ValueError("The given LHS compiled circuit is not known in this pipeline")
        if not self._compiler.has_symbolic(rhs_cc):
            raise ValueError("The given RHS compiled circuit is not known in this pipeline")
        lhs_sc = self._compiler.get_symbolic_circuit(lhs_cc)
        rhs_sc = self._compiler.get_symbolic_circuit(rhs_cc)
        prod_sc = SF.multiply(lhs_sc, rhs_sc, registry=self._op_registry)
        return self.compile(prod_sc)

    def differentiate(self, cc: CompiledCircuit, *, order: int = 1) -> CompiledCircuit:
        if not self._compiler.has_symbolic(cc):
            raise ValueError("The given compiled circuit is not known in this pipeline")
        if order <= 0:
            raise ValueError("The order of differentiation must be positive.")
        sc = self._compiler.get_symbolic_circuit(cc)
        diff_sc = SF.differentiate(sc, registry=self._op_registry, order=order)
        return self.compile(diff_sc)

    def conjugate(self, cc: CompiledCircuit) -> CompiledCircuit:
        if not self._compiler.has_symbolic(cc):
            raise ValueError("The given compiled circuit is not known in this pipeline")
        sc = self._compiler.get_symbolic_circuit(cc)
        conj_sc = SF.conjugate(sc, registry=self._op_registry)
        return self.compile(conj_sc)


def compile(sc: Circuit, ctx: PipelineContext | None = None) -> CompiledCircuit:
    if ctx is None:
        ctx = _PIPELINE_CONTEXT.get()
    return ctx.compile(sc)


def concatenate(*cc: CompiledCircuit, ctx: PipelineContext | None = None) -> CompiledCircuit:
    if ctx is None:
        ctx = _PIPELINE_CONTEXT.get()
    return ctx.concatenate(cc)


def integrate(
    cc: CompiledCircuit,
    scope: Iterable[int] | None = None,
    ctx: PipelineContext | None = None,
) -> CompiledCircuit:
    if ctx is None:
        ctx = _PIPELINE_CONTEXT.get()
    return ctx.integrate(cc, scope=scope)


def multiply(
    lhs_cc: CompiledCircuit, rhs_cc: CompiledCircuit, ctx: PipelineContext | None = None
) -> CompiledCircuit:
    if ctx is None:
        ctx = _PIPELINE_CONTEXT.get()
    return ctx.multiply(lhs_cc, rhs_cc)


def differentiate(
    cc: CompiledCircuit, ctx: PipelineContext | None = None, *, order: int = 1
) -> CompiledCircuit:
    if ctx is None:
        ctx = _PIPELINE_CONTEXT.get()
    return ctx.differentiate(cc, order=order)


def conjugate(cc: CompiledCircuit, ctx: PipelineContext | None = None) -> CompiledCircuit:
    if ctx is None:
        ctx = _PIPELINE_CONTEXT.get()
    return ctx.conjugate(cc)


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
