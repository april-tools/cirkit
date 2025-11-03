import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from cirkit.backend.torch.optimization.registry import (
    ParameterOptApplyFunc,
    ParameterOptMatch,
    ParameterOptPattern,
    ParameterOptPatternDefn,
)
from cirkit.backend.torch.parameters.nodes import (
    TorchFlattenParameter,
    TorchKroneckerParameter,
    TorchLogParameter,
    TorchLogSoftmaxParameter,
    TorchOuterProductParameter,
    TorchParameterNode,
    TorchReduceSumParameter,
    TorchSoftmaxParameter,
)
from cirkit.backend.torch.parameters.optimized import TorchEinsumParameter

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


class KroneckerOutParameterPattern(ParameterOptPatternDefn):
    """This pattern detects Kronecker parameter which are output of the graph.

    It is used when performing the tensor dot trick on sum or dot layers that have
    weights coming from such node.

    See [DenseKroneckerPattern][torch.cirkit.backend.optimization.layers.DenseKroneckerPattern].
    """

    @classmethod
    def is_output(cls) -> bool:
        return True

    @classmethod
    def entries(cls) -> list[type[TorchParameterNode]]:
        return [TorchKroneckerParameter]


class LogSoftmaxPattern(ParameterOptPatternDefn):
    """Detect a sequence of Softmax node -> Log node"""

    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> list[type[TorchParameterNode]]:
        return [TorchLogParameter, TorchSoftmaxParameter]


class ReduceSumOuterProductPattern(ParameterOptPatternDefn):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> list[type[TorchParameterNode]]:
        return [TorchReduceSumParameter, TorchOuterProductParameter]


def apply_log_softmax(  # pylint: disable=unused-argument
    compiler: "TorchCompiler", match: ParameterOptMatch
) -> tuple[TorchLogSoftmaxParameter]:
    """Fuse the log and softmax in one logsoftmax node.

    Args:
        compiler (TorchCompiler): The current compiler.
        match (ParameterOptMatch): The match object containing the modules to optimize.

    Returns:
       tuple[TorchLogSoftmaxParameter]: the corresponding logsoftmax node.
    """
    softmax = cast(TorchSoftmaxParameter, match.entries[1])
    log_softmax = TorchLogSoftmaxParameter(softmax.in_shapes[0], dim=softmax.dim)
    return (log_softmax,)


def _emit_outer_reduce_flatten_parameter(
    in_shape1: tuple[int, ...],
    in_shape2: tuple[int, ...],
    outer_dim: int,
    reduce_dim: int,
) -> tuple[TorchEinsumParameter] | tuple[TorchEinsumParameter, TorchFlattenParameter]:
    r"""Transform a reduce sum on the result of an outer product in a single einsum.

    The goal of this optimization is to reduce the memory usage by avoiding storing
    the outer product results, which can be quite heavy.

    For example: given matrices $A$ and $B$ of shapes $(b,i,j,l)$ and $(b,i,k,l)$.

    First case: we want `outer_dim=2` and `reduce_dim=2`, we would compute:
        1. The product: $bijl, bikl \rightarrow bijkl$ which makes a large matrices.
        2. Flatten $jk$ into $f$.
        3. The sum: $bifl \rightarrow bil$
    This computation can be done in a single einsum : $bijl, bikl \rightarrow bil$.

    Second case: we want `outer_dim=2` and `reduce_dim=3`, step 3 would look like:

    $bifl \rightarrow bif$

    Again, this can be done in a single einsum *and* a flatten operation:

    $bijl, bikl \rightarrow bijk$ and we then flatten $jk$ into $f$.


    Args:
        in_shape1 (tuple[int,...]): Shape of the first input **without** the fold dimension.
        in_shape2 (tuple[int,...]): Shape of the second input **without** the fold dimension
        outer_dim (int): The dimension used to compute the outer product.
        reduce_dim (int): The dimension used for the reduce sum.

    Returns:
        tuple[TorchEinsumParameter] | tuple[TorchEinsumParameter, TorchFlattenParameter]:
            Returns either a single einsum node if `outer_dim` and `reduce_dim` are equal,
            otherwise we need to flatten the two dimensions from the outer product.
    """
    # in_idx1 = [0, 1, 2, ..., N - 1]
    in_idx1: tuple[int, ...] = tuple(range(len(in_shape1)))
    # in_idx2 = [0, 1, 2, ..., N + 1, ..., N - 1]
    in_idx2: tuple[int, ...] = (
        tuple(range(outer_dim))
        + (len(in_shape1),)
        + tuple(range(outer_dim + 1, len(in_shape1)))
    )
    # Apply the reduction to the indices, as to get the output indices of the einsum
    reduce_idx: list[tuple[int] | tuple[int, int]] = (
        list((i,) for i in range(outer_dim))
        + [(outer_dim, len(in_shape1))]
        + list((i,) for i in range(outer_dim + 1, len(in_shape1)))
    )
    del reduce_idx[reduce_dim]
    out_idx: tuple[int, ...] = tuple(itertools.chain.from_iterable(reduce_idx))

    # If we are reducing the dimension along which we compute the Kronecker product,
    # we just need an einsum
    einsum = TorchEinsumParameter(
        (in_shape1, in_shape2), einsum=(in_idx1, in_idx2, out_idx)
    )
    if outer_dim == reduce_dim:
        return (einsum,)

    # If we are NOT reducing the dimension along which we compute the Kronecker product,
    # we need to flatten some dimensions after the einsum
    if reduce_dim < outer_dim:
        start_dim, end_dim = outer_dim - 1, outer_dim
    else:
        start_dim, end_dim = outer_dim, outer_dim + 1
    flatten = TorchFlattenParameter(einsum.shape, start_dim=start_dim, end_dim=end_dim)
    return einsum, flatten


def apply_sum_outer_prod_einsum(  # pylint: disable=unused-argument
    compiler: "TorchCompiler", match: ParameterOptMatch
) -> tuple[TorchEinsumParameter] | tuple[TorchEinsumParameter, TorchFlattenParameter]:
    """Transform the sum on an outer product into a single einsum to reduce memory usage.

    Args:
        compiler (TorchCompiler): Current torch compiler.
        match (ParameterOptMatch): Match containing the module to fuse.

    Returns:
        tuple[TorchEinsumParameter] | tuple[TorchEinsumParameter,TorchFlattenParameter]:
            returns the einsum corresponding to the matched modules.

    Raises:
        NotImplementedError: The function is not implemented for more than 4 dimensions.
    """
    outer_prod = cast(TorchOuterProductParameter, match.entries[1])
    reduce_sum = cast(TorchReduceSumParameter, match.entries[0])
    in_shape1, in_shape2 = outer_prod.in_shapes
    if len(in_shape1) > 4:
        raise NotImplementedError()
    outer_dim = outer_prod.dim
    reduce_dim = reduce_sum.dim
    return _emit_outer_reduce_flatten_parameter(
        in_shape1, in_shape2, outer_dim, reduce_dim
    )


DEFAULT_PARAMETER_OPT_RULES: Mapping[ParameterOptPattern, ParameterOptApplyFunc] = {
    LogSoftmaxPattern: apply_log_softmax,
    ReduceSumOuterProductPattern: apply_sum_outer_prod_einsum,
}
