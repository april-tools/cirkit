from typing import TYPE_CHECKING, Callable, Dict, cast

from cirkit.backend.torch.parameters.ops import (
    TorchCrossEinsumParameter,
    TorchLogParameter,
    TorchLogSoftmaxParameter,
    TorchOuterSumParameter,
    TorchReduceLSEParameter,
    TorchSoftmaxParameter,
)
from cirkit.backend.torch.parameters.parameter import (
    ParameterOptEntry,
    ParameterOptMatch,
    ParameterOptPattern,
    ParameterOptPatternDefn,
    TorchParameterNode,
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler

ParameterOptApplyFunc = Callable[[TorchCompiler, ParameterOptPattern], TorchParameterNode]


class LogSoftmaxPattern(ParameterOptPatternDefn):
    entries = {
        0: ParameterOptEntry(cls=TorchLogParameter),
        1: ParameterOptEntry(cls=TorchSoftmaxParameter),
    }


class ReduceLSEOuterSumPattern(ParameterOptPatternDefn):
    entries = {
        0: ParameterOptEntry(cls=TorchReduceLSEParameter),
        1: ParameterOptEntry(cls=TorchOuterSumParameter),
    }


def apply_log_softmax(
    compiler: "TorchCompiler", match: ParameterOptMatch
) -> TorchLogSoftmaxParameter:
    softmax = cast(TorchSoftmaxParameter, match.entries[1])
    return TorchLogSoftmaxParameter(softmax.in_shapes[0], dim=softmax.dim)


def apply_reduce_lse_outer_sum(
    compiler: "TorchCompiler", match: ParameterOptMatch
) -> TorchCrossEinsumParameter:
    reduce_lse = cast(TorchReduceLSEParameter, match.entries[0])
    outer_sum = cast(TorchOuterSumParameter, match.entries[1])
    return TorchCrossEinsumParameter(
        *outer_sum.in_shapes, outer_dim=outer_sum.dim, reduce_dim=reduce_lse.dim, lse_sum=True
    )


DEFAULT_PARAMETER_OPTIMIZATION_RULES: Dict[
    ParameterOptPattern, ParameterOptApplyFunc
] = {  # type: ignore[misc]
    LogSoftmaxPattern: apply_log_softmax,
    ReduceLSEOuterSumPattern: apply_reduce_lse_outer_sum,
}
