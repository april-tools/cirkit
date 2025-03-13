from collections.abc import Iterable
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from types import TracebackType

import cirkit.symbolic.functional as SF
from cirkit.backend.compiler import (
    SUPPORTED_BACKENDS,
    AbstractCompiler,
    CompiledCircuit,
    GateFunction,
    InitializerCompilationFunc,
    LayerCompilationFunc,
    ParameterCompilationFunc,
)
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import LayerOperator
from cirkit.symbolic.operators import LayerOperatorFunc
from cirkit.symbolic.registry import OperatorRegistry
from cirkit.utils.scope import Scope


class PipelineContext(AbstractContextManager):
    """A pipeline context is a Python context manager used to compile
    circuits and specify backend-specific compilation flags and optimizations.
    A pipeline context can also be used to register compilation rules for user-defined
    layers, parameterizations and initialization methods. Furthermore, new layer operators
    can be added to the context.
    """

    def __init__(self, backend: str = "torch", **backend_kwargs):
        """Initialzes a pipeline context, given the compilation backend and
            the compilation flags.

        Args:
            backend: The compilation backend.  The only backend supported is 'torch'.
            backend_kwargs: The compilation flags to pass to the compiler.

        Raises:
            ValuerError: if the compilation backend is unknown.
        """
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
        """Construts a pipeline context from the default backend.
            The default backend is 'torch'.

        Returns:
            A pipeline context from the default backend by specifying default
                values for the compilation flags.
        """
        return PipelineContext(backend="torch", semiring="lse-sum", fold=True, optimize=True)

    def __getitem__(self, sc: Circuit) -> CompiledCircuit:
        """Retrieves a compiled circuit, given a symbolic one.

        Args:
            sc: The symbolic circuit.

        Returns:
            The circuit compiled from the given symbolic circuit.
        """
        return self._compiler.get_compiled_circuit(sc)

    def __enter__(self) -> "PipelineContext":
        """Enters a pipeline context.

        Returns:
            Itself.
        """
        self._op_registry.__enter__()
        self._token = _PIPELINE_CONTEXT.set(self)
        return self

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        """Exit a pipeline context."""
        ret = self._op_registry.__exit__(__exc_type, __exc_value, __traceback)
        _PIPELINE_CONTEXT.reset(self._token)
        self._token = None
        return ret

    def add_operator_rule(self, op: LayerOperator, func: LayerOperatorFunc):
        """Add a new layer operator to the context.

        Args:
            op: The layer operator.
            func: The layer operator implementation.
        """
        self._op_registry.add_rule(op, func)

    def add_layer_compilation_rule(self, func: LayerCompilationFunc):
        """Add a new layer compilation rule to the current compilation backend.

        Args:
            func: The layer compilation rule.
        """
        self._compiler.add_layer_rule(func)

    def add_parameter_compilation_rule(self, func: ParameterCompilationFunc):
        """Add a new parameter compilation rule to the current compilation backend.

        Args:
            func: The parameter compilation rule.
        """
        self._compiler.add_parameter_rule(func)

    def add_initializer_compilation_rule(self, func: InitializerCompilationFunc):
        """Add a new initialization method compilation rule to the current compilation backend.

        Args:
            func: The initializer compilation rule.
        """
        self._compiler.add_initializer_rule(func)

    def compile(self, sc: Circuit) -> CompiledCircuit:
        """Compile a symbolic circuit.

        Args:
            sc: The symbolic circuit.

        Returns:
            A compiled circuit, whose type depends on the chosen compilation backend.
        """
        return self._compiler.compile(sc)

    def is_compiled(self, sc: Circuit) -> bool:
        """Check whether a symbolic circuit has been compiled in this context.

        Args:
            sc: The symbolic circuit.

        Returns:
            True if the given symbolic circuit has been compiled in this context, False otherwise.
        """
        return self._compiler.is_compiled(sc)

    def has_symbolic(self, cc: CompiledCircuit) -> bool:
        """Check whether a compiled circuit has a corresponding symbolic circuit
            in this context.

        Args:
            cc: The compiled circuit.

        Returns:
            False if the given compiled circuit has been compiled in this context, False otherwise.
        """
        return self._compiler.has_symbolic(cc)

    def get_compiled_circuit(self, sc: Circuit) -> CompiledCircuit:
        """Retrieves a compiled circuit, given a symbolic one.

        Args:
            sc: The symbolic circuit.

        Returns:
            The circuit compiled from the given symbolic circuit.
        """
        return self._compiler.get_compiled_circuit(sc)

    def get_symbolic_circuit(self, cc: CompiledCircuit) -> Circuit:
        """Retrieves a symbolic circuit, given a compiled one.

        Args:
            cc: The compiled circuit.

        Returns:
            The symbolic circuit associated to the given compiled one.
        """
        return self._compiler.get_symbolic_circuit(cc)

    def add_gate_function(self, name: str, function: GateFunction):
        """Register an external model implementation to the pipeline context.

        Args:
            name: The gate function name.
            function: The gate function object. For example, if using the torch backend, this can be
                an object of type [torch.nn.Module][torch.nn.Module].
        """
        self._compiler.add_gate_function(name, function)

    def get_gate_function(self, name: str) -> GateFunction:
        """Retrieves the gate function by its name.

        Args:
            name: The gate function name

        Returns:
            The external model object.
        """
        return self._compiler.get_gate_function(name)

    def concatenate(self, *cc: CompiledCircuit) -> CompiledCircuit:
        """Circuit concatenation interface for compiled circuits.
            See [concantenate][cirkit.symbolic.functional.concatenate] for more details.

        Args:
            *cc: A sequence of compiled circuits.

        Returns:
            The circuit that encodes the concatenation of the given circuits.

        Raises:
            ValueError: if the given circuits have not been compiled in this context.
        """
        for i, cci in enumerate(cc):
            if not self._compiler.has_symbolic(cci):
                raise ValueError(f"The {i}-th given compiled circuit is not known in this pipeline")
        sc = [self._compiler.get_symbolic_circuit(cci) for cci in cc]
        cat_sc = SF.concatenate(*sc, registry=self._op_registry)
        return self.compile(cat_sc)

    def integrate(self, cc: CompiledCircuit, scope: Scope | None = None) -> CompiledCircuit:
        """Circuit integration interface for compiled circuits.
            See [concantenate][cirkit.symbolic.functional.integrate] for more details.

        Args:
            cc: A compiled circuit.
            scope: The variables scope to integrate. If it is None, then all variables
                the given circuit is defined on are integrated.

        Returns:
            The circuit that encodes the integration over (some) variables of the given circuit.

        Raises:
            ValueError: if the given circuit has not been compiled in this context.
        """
        if not self._compiler.has_symbolic(cc):
            raise ValueError("The given compiled circuit is not known in this pipeline")
        sc = self._compiler.get_symbolic_circuit(cc)
        int_sc = SF.integrate(sc, scope=scope, registry=self._op_registry)
        return self.compile(int_sc)

    def multiply(self, cc1: CompiledCircuit, cc2: CompiledCircuit) -> CompiledCircuit:
        """Circuit multiplication interface for compiled circuits.
            See [multiply][cirkit.symbolic.functional.multiply] for more details.

        Args:
            cc1: The first compiled circuit.
            cc2: The second compiled circuit.

        Returns:
            The circuit that encodes the multiplicaton between the given circuits.

        Raises:
            ValueError: if the given circuits have not been compiled in this context.
        """
        if not self._compiler.has_symbolic(cc1):
            raise ValueError("The first compiled circuit is not known in this pipeline")
        if not self._compiler.has_symbolic(cc2):
            raise ValueError("The second compiled circuit is not known in this pipeline")
        sc1 = self._compiler.get_symbolic_circuit(cc1)
        sc2 = self._compiler.get_symbolic_circuit(cc2)
        prod_sc = SF.multiply(sc1, sc2, registry=self._op_registry)
        return self.compile(prod_sc)

    def differentiate(self, cc: CompiledCircuit, *, order: int = 1) -> CompiledCircuit:
        """Circuit differentiation interface for compiled circuits.
            See [differentiate][cirkit.symbolic.functional.differentiate] for more details.

        Args:
            cc: The compiled circuit.
            order: The differentiation order.

        Returns:
            The circuit that encodes the differentiation of the given compiled circuit.

        Raises:
            ValueError: if the given circuit has not been compiled in this context.
            ValueError: if the differentiation order is not a positive integer.
        """
        if not self._compiler.has_symbolic(cc):
            raise ValueError("The given compiled circuit is not known in this pipeline")
        if order <= 0:
            raise ValueError("The order of differentiation must be positive.")
        sc = self._compiler.get_symbolic_circuit(cc)
        diff_sc = SF.differentiate(sc, order=order, registry=self._op_registry)
        return self.compile(diff_sc)

    def conjugate(self, cc: CompiledCircuit) -> CompiledCircuit:
        """Circuit conjugation interface for compiled circuits.
            See [conjugate][cirkit.symbolic.functional.conjugate] for more details.

        Args:
            cc: The compiled circuit.

        Returns:
            The circuit that encodes the complex conjugation of the given compiled circuit.

        Raises:
            ValueError: if the given circuit has not been compiled in this context.
        """
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
    cc1: CompiledCircuit, cc2: CompiledCircuit, ctx: PipelineContext | None = None
) -> CompiledCircuit:
    if ctx is None:
        ctx = _PIPELINE_CONTEXT.get()
    return ctx.multiply(cc1, cc2)


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
        # pylint: disable-next=import-outside-toplevel
        from cirkit.backend.torch.compiler import TorchCompiler

        return TorchCompiler(**backend_kwargs)
    assert False


# Context variable holding the current global pipeline.
# This is updated when entering a pipeline context.
_PIPELINE_CONTEXT: ContextVar[PipelineContext] = ContextVar(
    "_PIPELINE_CONTEXT", default=PipelineContext.from_default_backend()
)
