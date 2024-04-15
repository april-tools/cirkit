from typing import Optional, Dict, Any

import torch
from torch import Tensor

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.layers import InputLayer
from cirkit.backend.torch.reparams import Reparameterization, ExpReparam, LeafReparam
from cirkit.pipeline import PipelineContext
from cirkit.symbolic.functional import integrate, multiply, differentiate
from cirkit.symbolic.sym_circuit import SymCircuit, SymCircuitOperator
from cirkit.symbolic.sym_layers import SymCategoricalLayer, SymDenseLayer, SymKroneckerLayer, SymMixingLayer, \
    SymInputLayer, SymLayerOperation, SymLayerOperator
from cirkit.symbolic.sym_params import SymSoftplus, SymParameter, SymSoftmax, AbstractSymParameter, SymKronecker, \
    SymParameterPlaceholder
from cirkit.templates.region_graph.algorithms import FullyFactorized, QuadTree
from cirkit.utils import Scope


def categorical_layer_factory(
        scope: Scope,
        num_units: int,
        num_channels: int
) -> SymCategoricalLayer:
    # Q: Where do we specify a particular parameterizations?
    # Parameterizations are not needed in the symbolic data structure, but
    # where should they be specified such that they are also extensible?
    # (same question for all kind of layers)
    return SymCategoricalLayer(scope, num_units, num_channels, num_categories=256)


def dense_layer_factory(scope: Scope, num_input_units: int, num_output_units: int) -> SymDenseLayer:
    return SymDenseLayer(scope, num_input_units, num_output_units)


def kronecker_layer_factory(scope: Scope, num_input_units: int, arity: int) -> SymKroneckerLayer:
    return SymKroneckerLayer(scope, num_input_units, arity)


def simplest_unnormalized_pc() -> None:
    # Create a region graph
    qt = QuadTree(shape=(28, 28), struct_decomp=True)

    # Instantiate a symbolic circuit from the region graph,
    # by specifying factories that construct symbolic layers
    sc = SymCircuit.from_region_graph(
        qt, num_input_units=8, num_sum_units=16,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory
    )

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

    # # # # # # # # # # # # # # # # # # # # # #
    # Low-level APIs
    # - The user is happy to operate directly on symbolic circuits
    # - The user might need to implement new operators over layers/circuits
    # - The user wants to specify a full symbolic pipeline first, perhaps save/plot it, and then compile it
    int_sc = integrate(sc)        # Explicitly use the symbolic functional APIs
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
    p = SymCircuit.from_region_graph(
        qt, num_input_units=8, num_sum_units=16,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory
    )
    q = SymCircuit.from_region_graph(
        ff, num_input_units=12, num_sum_units=1,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory
    )

    # Create a pipeline context
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
    r = differentiate(q)
    s = multiply(p, r)
    t = integrate(s)
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
    class SymPolyGaussianLayer(SymInputLayer):
        def __init__(
            self,
            scope: Scope,
            num_output_units: int,
            num_channels: int = 1,
            operation: Optional[SymLayerOperation] = None,
            degree: int = 2,
            coeffs: Optional[SymParameter] = None
        ):
            super().__init__(scope, num_output_units, num_channels=num_channels, operation=operation)
            self.degree = degree
            if coeffs is None:
                coeffs = SymParameter(self.num_variables, num_channels, num_output_units)
            self.coeffs = coeffs

        @property
        def hparams(self) -> Dict[str, Any]:
            hparams = super().hparams
            hparams.update(degree=self.degree)
            return hparams

        @property
        def learnable_params(self) -> Dict[str, AbstractSymParameter]:
            return dict(coeffs=self.coeffs)

    # Then, the user must write the actual layer for some backend of choice, e.g., torch
    # Important: note that the learnable parameters 'coeffs' are passed as arguments to the initializer
    class PolyGaussianLayer(InputLayer):
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
    ) -> SymPolyGaussianLayer:
        return SymPolyGaussianLayer(scope, num_units, num_channels, degree=3)

    # With these symbolic classes, we are ready to instantiate a couple of symbolic circuits
    p = SymCircuit.from_region_graph(
        qt, num_input_units=16, num_sum_units=16,
        input_factory=poly_gaussian_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory
    )
    q = SymCircuit.from_region_graph(
        qt, num_input_units=16, num_sum_units=16,
        input_factory=poly_gaussian_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory
    )

    ################################ PART 2 ################################

    # Now, we need to extend the compiler such that it knows that:
    # - SymPolyGaussianLayer must be compiled into PolyGaussianLayer
    # In order to do so, we write a compilation rule
    def compile_poly_gaussian(sl: SymPolyGaussianLayer, compiler: TorchCompiler) -> PolyGaussianLayer:
        # Q: When using operators (e.g., product of poly gaussian) should the compilation rule
        #    detect the kind of operator and construct the 'coeffs' parameterization accordingly?
        #    In short, should the compilation rule be responsible for parameter sharing?
        #    If yes, how to write the APIs such that this is as painless as possible?
        # I think it is fair to make the compilation rule deal with parameter sharing.
        # That is, if this poly-gaussian is the result of a product, then it should provide the information on
        # how to compile the correct parameterization as to encode the kronecker product of the two input 'coeffs'.
        # Important: the symbolic layer data structure already contains the information to make this work.
        return PolyGaussianLayer(
            sl.num_variables, sl.num_channels, sl.num_output_units,
            degree=sl.degree, coeffs=ExpReparam(LeafReparam())
        )

    # We extend the pipeline context by registering new compilation rules
    ctx = PipelineContext(backend='torch', fold=True, einsum=True)
    ctx.register_compilation_rule(compile_poly_gaussian)
    tp = ctx.compile(p)  # These compilations will call our custom compilation rule
    tq = ctx.compile(q)

    ################################ PART 2 ################################

    # Let's suppose the user want to implement the product between their new layer.
    # As usual, it must implement first the symbolic protocol for that.
    # Such symbolic protocol is the symbolic product operator for such layer.
    def multiply_poly_gaussian(
        lhs: SymPolyGaussianLayer, rhs: SymPolyGaussianLayer
    ) -> SymPolyGaussianLayer:
        assert lhs.scope == rhs.scope
        assert lhs.num_channels == rhs.num_channels
        return SymPolyGaussianLayer(
            lhs.scope,
            lhs.num_output_units * rhs.num_output_units,
            num_channels=lhs.num_channels,
            operation=SymLayerOperation(  # Record the product operation and operands
                SymLayerOperator.KRONECKER,
                operands=(lhs, rhs)
            ),
            degree=lhs.degree * rhs.degree + 1,  # Record the change in hyperparameters or other static attributes ...
            coeffs=SymKronecker(  # ... as well as the change over the learnable parameters
                # Important: use place-holders to keep track of parameter sharing
                SymParameterPlaceholder(opd_id=0, name='coeffs'),
                SymParameterPlaceholder(opd_id=1, name='coeffs')
            )  # Each placeholder will tell the compiler to re-use the parameters named 'coeffs'
               # that are kept in the operands 0 and 1 (see SymLayerOperation specification above)
        )

    # # # # # # # # # # # # # # # # # # # # # #
    # High-level APIs
    # To use the above symbolic product protocol, we have to register it as a symbolic operator
    ctx.register_operator_rule(SymCircuitOperator.MULTIPLICATION, multiply_poly_gaussian)
    tr = ctx.multiply(tp, tq)  # This will call ctx.compile(product(p, q)) internally

    # # # # # # # # # # # # # # # # # # # # # #
    # Lower-level APIs (i.e., explicit manipulation of symbolic circuits and uses the symbolic functional APIs)
    with ctx:
        r = multiply(p, q)  # No need to specify anything here, the with statement will take care of the new protocols
    tr = ctx.compile(r)  # We do not need to be inside the context for compilation, as the protocol above is symbolic
    # Important: note that outside the with statement, multiply will raise a SymbolicOperatorNotFound exception


# def example_serialization() -> None:
#     ################################ PART 1 ################################
#
#     # Create two region graphs
#     qt = QuadTree(shape=(28, 28), struct_decomp=True)
#     ff = FullyFactorized(num_vars=28 * 28)
#
#     # Instantiate two symbolic circuits from the region graphs,
#     # by specifying symbolic layers and symbolic parameterizations
#     p = SymCircuit.from_region_graph(
#         qt,
#         num_input_units=16,
#         num_sum_units=16,
#         input_cls=SymCategoricalLayer,
#         sum_cls=SymSumLayer,
#         prod_cls=SymHadamardLayer,
#         input_param_cls=SymSoftmax,
#         sum_param_cls=SymSoftplus,
#     )
#     q = SymCircuit.from_region_graph(
#         ff,
#         num_input_units=16,
#         input_cls=SymCategoricalLayer,
#         prod_cls=SymHadamardLayer,
#         input_param_cls=SymSoftmax,
#     )
#
#     # Construct the pipeline
#     r = differentiate(q)
#     s = multiply(p, r)
#
#     # Create a pipeline context for compilation
#     # Given the output of a computational graph over circuits (i.e., the root of the DAG),
#     # ctx.compile compiles all the circuits in topological ordering (at least by default)
#     # The backend can be specified with a flag, and available optimizations using kwargs.
#     # For instance, the backend 'torch' might support the following optimizations:
#     # - fold=True  enables folding;
#     # - einsum=True  suggests compiling to layers using einsums when possible.
#     ctx = PipelineContext(backend="torch")
#     ctx.compile(s, fold=True, einsum=True)
#
#     # Retrieve the compiled tensorized circuits from the pipeline context
#     # These circuits are torch modules and share parameters accordingly
#     tensorized_r, tensorized_s = ctx[r], ctx[s]
#
#     # Do inference or learning with the compiled circuits
#     ...
#
#     ################################ PART 2 ################################
#
#     # Let's suppose we now want to save this pipeline and load it later
#     # Saving a single symbolic/tensorized circuit would not allow us to do other operations later,
#     # unless they are the input circuits of the pipeline. This is because we have parameter sharing
#     # between circuits, and connections between symbolic representations.
#     # For this reason, we can save the pipeline context, which already knows the backed and therefore
#     # will use the relevant save routines for that backend.
#     # Note that we can save symbolic and tensorized circuits in different files.
#     ctx.save("symbolic-pipeline.ckit", "tensorized-circuits.pt")
#
#     # Now, let's load again the context
#     ctx = PipelineContext.load("symbolic-pipeline.ckit", "tensorized-circuits.pt")
#
#     # Since we have saved SymCircuitthe context, we can incrementally operate on circuits and compile them,
#     # without losing any useful data structure from previous compilations (e.g., references between layers)
#     # That is, we can extend the symbolic pipeline using other operators
#     s = ctx[
#         "product.0"
#     ]  # Let's assume symbolic circuits will have labels or names that the user can set
#     t = integrate(s)
#     ctx.compile(t)
#
#     # Retrieve the compiled tensorized circuit from the pipeline context
#     tensorized_t = ctx[t]
#
#     # Do inference or learning with the compiled circuits
#     ...
