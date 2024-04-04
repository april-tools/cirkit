# Templates imports
from cirkit.templates.region_graph.algorithms import QuadTree

# Symbolic stuff imports
from cirkit.symbolic.symb_layers import *
from cirkit.symbolic.symb_params import *
from cirkit.symbolic.symb_circuit import SymbCircuit
from cirkit.symbolic.functional import integrate

# Tensorized stuff imports
from cirkit.tensorized.compilers import PipelineContext


def example_simple_pc() -> None:
    # Create a region graph
    qt = QuadTree(shape=(28, 28), struct_decomp=True)

    # Instantiate a symbolic circuit from the region graph,
    # by specifying symbolic layers and symbolic parameterizations
    sc = SymbCircuit.from_region_graph(
        qt, num_input_units=16, num_channels=16, num_classes=10,
        input_cls=SymbCategoricalLayer, sum_cls=SymbSumLayer, prod_cls=SymbKroneckerLayer,
        input_param_cls=SymbSoftmax, sum_param_cls=SymbSoftmax
    )

    # Integrate the circuit symbolically (sum over all variables if not otherwise specified)
    int_sc = integrate(sc)

    # Create a pipeline context for compilation
    # Given the output of a computational graph over circuits (i.e., the root of the DAG),
    # ctx.compile compiles all the circuits in topological ordering (at least by default)
    # The backend can be specified with a flag, and available optimizations using kwargs.
    # For instance, the backend 'torch' might support the following optimizations:
    # - fold=True  enables folding;
    # - einsum=True  suggests compiling to layers using einsums when possible.
    ctx = PipelineContext()
    ctx.compile(int_sc, backend='torch', fold=True, einsum=True)

    # Retrieve the compiled tensorized circuits from the pipeline context
    # These circuits are torch modules and share parameters accordingly
    tc, int_tc = ctx[sc], ctx[int_sc]

    # Do inference or learning with the compiled circuits
    ...
