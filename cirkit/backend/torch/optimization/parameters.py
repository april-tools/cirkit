# pylint: disable=bad-mcs-classmethod-argument

import itertools
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


class KroneckerOutParameterPattern(ParameterOptPattern):
    @classmethod
    def is_output(cls) -> bool:
        return True

    @classmethod
    def entries(cls) -> list[type[TorchParameterNode]]:
        return [TorchKroneckerParameter]


class LogSoftmaxPattern(ParameterOptPatternDefn):
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
    softmax = cast(TorchSoftmaxParameter, match.entries[1])
    log_softmax = TorchLogSoftmaxParameter(softmax.in_shapes[0], dim=softmax.dim)
    return (log_softmax,)


def _emit_outer_reduce_flatten_parameter(
    in_shape1: tuple[int, ...], in_shape2: tuple[int, ...], outer_dim: int, reduce_dim: int
) -> tuple[TorchEinsumParameter] | tuple[TorchEinsumParameter, TorchFlattenParameter]:
    # in_idx1 = [0, 1, 2, ..., N - 1]
    in_idx1: tuple[int, ...] = tuple(range(len(in_shape1)))
    # in_idx2 = [0, 1, 2, ..., N + 1, ..., N - 1]
    in_idx2: tuple[int, ...] = (
        tuple(range(outer_dim)) + (len(in_shape1),) + tuple(range(outer_dim + 1, len(in_shape1)))
    )
    # Apply the reduction to the indices, as to get the output indices of the einsum
    reduce_idx: list[tuple[int, ...]] = (
        list((i,) for i in range(outer_dim))
        + [(outer_dim, len(in_shape1))]
        + list((i,) for i in range(outer_dim + 1, len(in_shape1)))
    )
    del reduce_idx[reduce_dim]
    out_idx: tuple[int, ...] = tuple(itertools.chain.from_iterable(reduce_idx))

    # If we are reducing the dimension along which we compute the Kronecker product,
    # we just need an einsum
    einsum = TorchEinsumParameter((in_shape1, in_shape2), einsum=(in_idx1, in_idx2, out_idx))
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
    outer_prod = cast(TorchOuterProductParameter, match.entries[1])
    reduce_sum = cast(TorchReduceSumParameter, match.entries[0])
    in_shape1, in_shape2 = outer_prod.in_shapes
    if len(in_shape1) > 4:
        raise NotImplementedError()
    outer_dim = outer_prod.dim
    reduce_dim = reduce_sum.dim
    return _emit_outer_reduce_flatten_parameter(in_shape1, in_shape2, outer_dim, reduce_dim)


# pylint: disable-next=line-too-long
DEFAULT_PARAMETER_OPT_RULES: dict[ParameterOptPattern, ParameterOptApplyFunc] = {  # type: ignore[misc]
    LogSoftmaxPattern: apply_log_softmax,
    ReduceSumOuterProductPattern: apply_sum_outer_prod_einsum,
}
