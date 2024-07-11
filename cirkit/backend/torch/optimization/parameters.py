from typing import TYPE_CHECKING, Dict, cast

from cirkit.backend.torch.optimization.registry import (
    ParameterOptApplyFunc,
    ParameterOptMatch,
    ParameterOptPattern,
    ParameterOptPatternDefn,
)
from cirkit.backend.torch.parameters.ops import (
    TorchLogParameter,
    TorchLogSoftmaxParameter,
    TorchOuterSumParameter,
    TorchReduceLSEParameter,
    TorchSoftmaxParameter,
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


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
