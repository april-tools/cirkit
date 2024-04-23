from typing import Optional, Dict, Any

import torch
from torch import Tensor

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.layers import TorchInputLayer
from cirkit.backend.torch.reparams import Reparameterization
from cirkit.pipeline import PipelineContext, compile, integrate, multiply, differentiate
import cirkit.symbolic.functional as SF
from cirkit.symbolic.circuit import Circuit, CircuitBlock
from cirkit.symbolic.layers import CategoricalLayer, DenseLayer, KroneckerLayer, InputLayer, \
    LayerOperation, MixingLayer, PlaceholderParameter
from cirkit.symbolic.params import SoftmaxParameter, AbstractParameter, Parameter, Parameterization, KroneckerParameter
from cirkit.templates.region_graph.algorithms import FullyFactorized, QuadTree
from cirkit.utils.scope import Scope


def categorical_layer_factory(
        scope: Scope,
        num_units: int,
        num_channels: int
) -> CategoricalLayer:
    return CategoricalLayer(scope, num_units, num_channels, num_categories=256)


def dense_layer_factory(scope: Scope, num_input_units: int, num_output_units: int) -> DenseLayer:
    return DenseLayer(scope, num_input_units, num_output_units, weight_param=lambda w: SoftmaxParameter(w))


def mixing_layer_factory(scope: Scope, num_units: int, arity: int) -> MixingLayer:
    return MixingLayer(scope, num_units, arity, weight_param=lambda w: SoftmaxParameter(w))


def kronecker_layer_factory(scope: Scope, num_input_units: int, arity: int) -> KroneckerLayer:
    return KroneckerLayer(scope, num_input_units, arity)


def simplest_unnormalized_pc() -> None:
    # Create a region graph
    qt = QuadTree(shape=(28, 28), struct_decomp=True)

    # Instantiate a symbolic circuit from the region graph,
    # by specifying factories that construct symbolic layers
    sc = Circuit.from_region_graph(
        qt, num_input_units=8, num_sum_units=16,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory,
        mixing_factory=mixing_layer_factory
    )

    # # # # # # # # # # # # # # # # # # # # # #
    # Singleton-level APIs
    # - The user does not want to set upt a pipeline context explicitly
    # - The user wants to use the default backend (i.e., torch)
    tc = compile(sc)
    int_tc = integrate(tc)

    # # # # # # # # # # # # # # # # # # # # # #
    # Context-level APIs
    # - The user wants to set up the pipeline context with a different backend and/or flags
    #
    # Given the output of a computational graph over circuits (i.e., the root of the DAG),
    # ctx.compile compiles all the circuits in topological ordering (at least by default)
    # The backend can be specified with a flag, and available optimizations using kwargs.
    # For instance, the backend 'torch' might support the following optimizations:
    # - fold=True  enables folding;
    # - einsum=True  suggests compiling to layers using einsums when possible.
    # Also, the context stores the symbolic operator definitions, which might be extended (see examples below)
    ctx = PipelineContext(backend='torch', fold=True, einsum=True)

    # # # # # # # # # # # # # # # # # # # # # #
    # High-level APIs (e.g., for practitioners)
    # - The user does not want to manipulate symbolic circuits explicitly
    # - The user does not need to implement new layers or new operators over layers/circuits, i.e.,
    #   they will only use what already present in the library in terms of layers and backends
    tc = ctx.compile(sc)        # Apart from the inputs to the pipeline, the user does not touch symbolic circuits
    int_tc = ctx.integrate(tc)  # This is equivalent to 'ctx.compile(integrate(sc))' (i.e., greedy compilation)
    # TODO: remove ctx?

    # # # # # # # # # # # # # # # # # # # # # #
    # Low-level APIs
    # - The user is happy to operate directly on symbolic circuits
    # - The user might need to implement new operators over layers/circuits
    # - The user wants to specify a full symbolic pipeline first, perhaps save/plot it, and then compile it
    int_sc = SF.integrate(sc)     # Explicitly use the symbolic functional APIs
    int_tc = ctx.compile(int_sc)  # Compiling the root of the pipeline implies compiling the other circuits too
    tc = ctx[sc]                  # Retrieve the compiled circuit corresponding to the input circuit to the pipeline

    # Do learning/inference with the compiled circuits
    ...


def complex_pipeline() -> None:
    # Create two region graphs
    qt = QuadTree(shape=(28, 28), struct_decomp=True)
    ff = FullyFactorized(num_vars=28 * 28)

    # Instantiate two symbolic circuits from the region graphs,
    # by specifying symbolic layers and symbolic parameterizations
    p = Circuit.from_region_graph(
        qt, num_input_units=8, num_sum_units=16,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory,
        mixing_factory=mixing_layer_factory
    )
    q = Circuit.from_region_graph(
        ff, num_input_units=12, num_sum_units=1,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory,
        mixing_factory=mixing_layer_factory
    )

    # # # # # # # # # # # # # # # # # # # # # #
    # Singleton-level APIs
    # - The user does not want to set upt a pipeline context explicitly
    # - The user wants to use the default backend (i.e., torch)
    # - The user does not want to extend the operators over layers/circuits, nor define compilation rules
    tp, tq = compile(p), compile(q)
    tr = differentiate(tq)
    ts = multiply(tp, tr)
    tt = integrate(ts)  # These are all already-compiled circuits

    # # # # # # # # # # # # # # # # # # # # # #
    # Context-level APIs
    # - The user wants to set up the pipeline context with a different backend and/or flags
    #
    ctx = PipelineContext(backend='torch', fold=True, einsum=True)

    # # # # # # # # # # # # # # # # # # # # # #
    # High-level API (e.g., for practitioners, as explained above)
    # - The user explicitly operates only on compiled circuit representations
    # Internally, the pipeline interleaves symbolic operations and compilation steps
    tp, tq = ctx.compile(p), ctx.compile(q)
    tr = ctx.differentiate(tq)
    ts = ctx.multiply(tp, tr)
    tt = ctx.integrate(ts)  # These are all already-compiled circuits

    # # # # # # # # # # # # # # # # # # # # # #
    # Low-level API (as explained above)
    # - The user explicitly operates on symbolic circuit representations, then it compile the pipeline
    # - This is for advanced users who want to extend the domain specific language, e.g.,
    #   add new layers or new layer/circuit symbolic operators (see examples below about this)
    r = SF.differentiate(q)
    s = SF.multiply(p, r)
    t = SF.integrate(s)
    tt = ctx.compile(t)  # This also compiles the other circuits in the pipeline
    tr = ctx[r]  # Retrieve other compiled circuits from the pipeline context

    # Do learning/inference with the compiled circuits
    ...


def lib_extension() -> None:
    ################################ PART 1 ################################
    # Create a region graph
    qt = QuadTree(shape=(28, 28), struct_decomp=True)

    # Let's suppose the user wants to introduce a new input layer called 'PolyGaussian'.
    # Ideally, they have to first create a symbolic token for it,
    # with the hyperparameters and parameters they care about.
    class PolyGaussianLayer(InputLayer):
        def __init__(
            self,
            scope: Scope,
            num_output_units: int,
            num_channels: int = 1,
            degree: int = 2,
            coeffs: Optional[AbstractParameter] = None,
            coeffs_param: Optional[Parameterization] = None
        ):
            super().__init__(scope, num_output_units, num_channels)
            self.degree = degree
            if coeffs is None:
                coeffs = Parameter(self.num_variables, num_channels, num_output_units)
            if coeffs_param is not None:
                coeffs = coeffs_param(coeffs)
            self.coeffs = coeffs

        @property
        def hparams(self) -> Dict[str, Any]:
            hparams = super().hparams
            hparams.update(degree=self.degree)
            return hparams

        @property
        def learnable_params(self) -> Dict[str, AbstractParameter]:
            return dict(coeffs=self.coeffs)

    # Then, the user must write the actual layer for some backend of choice, e.g., torch
    # Important: note that the learnable parameters 'coeffs' are passed as arguments to the initializer
    class TorchPolyGaussianLayer(TorchInputLayer):
        def __init__(
            self,
            num_vars: int,
            num_channels: int,
            num_output_units: int,
            *,
            degree: int,
            coeffs: Reparameterization
        ):
            super().__init__(
                num_input_units=num_channels,
                num_output_units=num_output_units,
                arity=num_vars
            )
            self.degree = degree
            self.coeffs = coeffs

        def polyval(self, p: Tensor, x: Tensor) -> Tensor:
            ...

        def forward(self, x: Tensor) -> Tensor:
            params_exp, params_poly = self.weight()
            # The exponent is always in log-space.
            gaussian = self.polyval(self.params_exp, x)
            if params_poly is None:
                return gaussian
            factor = self.polyval(self.params_poly, x)
            return torch.mul(gaussian, factor)

    # Next, the user must write some factory for it (or use a lambda function)
    def poly_gaussian_factory(
        scope: Scope,
        num_units: int,
        num_channels: int
    ) -> PolyGaussianLayer:
        return PolyGaussianLayer(scope, num_units, num_channels, degree=3)

    # With these symbolic classes, we are ready to instantiate a couple of symbolic circuits
    p = Circuit.from_region_graph(
        qt, num_input_units=16, num_sum_units=16,
        input_factory=poly_gaussian_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory,
        mixing_factory=mixing_layer_factory
    )
    q = Circuit.from_region_graph(
        qt, num_input_units=16, num_sum_units=16,
        input_factory=poly_gaussian_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory,
        mixing_factory=mixing_layer_factory
    )

    ################################ PART 2 ################################

    # Now, we need to extend the compiler such that it knows that:
    # - SymPolyGaussianLayer must be compiled into PolyGaussianLayer
    # In order to do so, we write a compilation rule
    def compile_poly_gaussian(sl: PolyGaussianLayer, compiler: TorchCompiler) -> TorchPolyGaussianLayer:
        return TorchPolyGaussianLayer(
            sl.num_variables, sl.num_channels, sl.num_output_units,
            degree=sl.degree,
            # The Torch compiler comes with a utility method for compiling symbolic parameters
            # In other words, the compiler is responsible for parameter sharing
            # E.g., 'coeffs' is the kronecker product of other 'coeffs' (see below)
            coeffs=compiler.compile_learnable_parameter(sl.coeffs)
        )

    # # # # # # # # # # # # # # # # # # # # # #
    # Context-level APIs
    # We extend the pipeline context by registering new compilation rules
    ctx = PipelineContext(backend='torch', fold=True, einsum=True)
    ctx.add_layer_compilation_rule(compile_poly_gaussian)
    tp = ctx.compile(p)  # These compilations will call our custom compilation rule
    tq = ctx.compile(q)

    # Let's suppose the user wants to implement the product operation of circuits with their new layer.
    # As usual, it must implement first the symbolic protocol for that.
    # Such symbolic protocol is the symbolic product operator for such a layer.
    def multiply_poly_gaussian(
        lhs: PolyGaussianLayer, rhs: PolyGaussianLayer
    ) -> CircuitBlock:
        assert lhs.scope == rhs.scope
        assert lhs.num_channels == rhs.num_channels
        sl = PolyGaussianLayer(
            lhs.scope,
            num_output_units=lhs.num_output_units * rhs.num_output_units,
            num_channels=lhs.num_channels,
            degree=lhs.degree * rhs.degree + 1,  # Record the change in hyperparameters or other static attributes ...
            coeffs=KroneckerParameter(  # ... as well as the change over the learnable parameters
                # Each placeholder will "tell the compiler" to re-use the parameters named 'coeffs',
                # which are stored (or referenced elsewhere) in the operand layers 'lhs' and 'rhs'
                PlaceholderParameter(lhs, name='coeffs'),
                PlaceholderParameter(rhs, name='coeffs')
            )
        )
        return CircuitBlock.from_layer(sl)

    # # # # # # # # # # # # # # # # # # # # # #
    # High-level APIs
    # To use the above symbolic product protocol, we have to register it as a symbolic operator
    ctx.register_operator_rule(LayerOperation.MULTIPLICATION, multiply_poly_gaussian)
    tr = ctx.multiply(tp, tq)  # This will call ctx.compile(product(p, q)) internally

    # # # # # # # # # # # # # # # # # # # # # #
    # Lower-level APIs (i.e., explicit manipulation of symbolic circuits and uses the symbolic functional APIs)
    ctx.register_operator_rule(LayerOperation.MULTIPLICATION, multiply_poly_gaussian)
    with ctx:
        r = SF.multiply(p, q)  # No need to specify anything here, the with statement will take care of the new protocols
    tr = ctx.compile(r)  # We do not need to be inside the context for compilation, as the protocol above is symbolic
    # Important: note that outside the with statement, multiply will raise a SymbolicOperatorNotFound exception


def serialization() -> None:
    # Create two region graphs
    qt = QuadTree(shape=(28, 28), struct_decomp=True)
    ff = FullyFactorized(num_vars=28 * 28)

    # Instantiate two symbolic circuits from the region graphs,
    # by specifying symbolic layers and symbolic parameterizations
    p = Circuit.from_region_graph(
        qt,
        num_input_units=16,
        num_sum_units=16,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory,
        mixing_factory=mixing_layer_factory
    )
    q = Circuit.from_region_graph(
        ff,
        num_input_units=16,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory,
        mixing_factory=mixing_layer_factory
    )

    # Construct the pipeline
    ctx = PipelineContext(backend='torch', fold=True, einsum=True)
    tp = ctx.compile(p)
    tq = ctx.compile(q)
    tr = ctx.differentiate(tq)
    ts = ctx.multiply(tp, tr)

    # Let's suppose we now want to save this pipeline and load it later
    # Saving a single symbolic/tensorized circuit would not allow us to do other operations later,
    # unless they are the input circuits of the pipeline. This is because we have parameter sharing
    # between circuits, and also connections between symbolic representations.
    # For this reason, we save the pipeline context, which already knows the backed and therefore
    # will use the relevant save/load routines available for that backend.
    ctx.save('symbolic-pipeline.ckit', 'tensorized-circuits.pt')

    # Now, let's load again the context
    # The following will:
    # (i)   load the symbolic representations of all the circuits in the pipeline
    # (ii)  compile them using the backend specified above
    # (iii) load the state (i.e., the parameters and buffers) of the compiled circuits
    ctx = PipelineContext.load('symbolic-pipeline.ckit', 'tensorized-circuits.pt')

    # Since we saved the entire pipeline, we can incrementally operate on circuits and compile them,
    # without losing any useful data structure from previous compilations (e.g., references between layers)
    # That is, we can extend the pipeline using other operations
    # Let's assume symbolic circuits will have labels (either set by the user or automatically set by the operations)
    s = ctx['product.0']
    tt = ctx.integrate(ts)

    # Do inference or learning with the compiled circuits
    ...
