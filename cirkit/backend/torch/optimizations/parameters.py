from typing import TYPE_CHECKING, Callable, Dict, Type, cast

from cirkit.backend.torch.graph.optimize import GraphOptMatch, GraphOptPatternDefn
from cirkit.backend.torch.parameters.ops import (
    TorchLogParameter,
    TorchLogSoftmaxParameter,
    TorchOuterSumParameter,
    TorchReduceLSEParameter,
    TorchSoftmaxParameter,
)
from cirkit.backend.torch.parameters.parameter import TorchParameterNode

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler

ParameterOptPatternDefn = GraphOptPatternDefn[TorchParameterNode]

ParameterOptPattern = Type[ParameterOptPatternDefn]

ParameterOptMatch = GraphOptMatch[TorchParameterNode]

ParameterOptApplyFunc = Callable[["TorchCompiler", ParameterOptPattern], TorchParameterNode]


class LogSoftmaxPattern(ParameterOptPatternDefn):
    entries = {
        0: TorchLogParameter,
        1: TorchSoftmaxParameter,
    }


class ReduceLSEOuterSumPattern(ParameterOptPatternDefn):
    entries = {
        0: TorchReduceLSEParameter,
        1: TorchOuterSumParameter,
    }


def apply_log_softmax(
    compiler: "TorchCompiler", match: ParameterOptMatch
) -> TorchLogSoftmaxParameter:
    softmax = cast(TorchSoftmaxParameter, match.entries[1])
    return TorchLogSoftmaxParameter(softmax.in_shapes[0], dim=softmax.dim)


DEFAULT_PARAMETER_OPT_RULES: Dict[
    ParameterOptPattern, ParameterOptApplyFunc
] = {  # type: ignore[misc]
    LogSoftmaxPattern: apply_log_softmax
}
