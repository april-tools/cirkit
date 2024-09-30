from typing import TYPE_CHECKING, cast

from cirkit.backend.torch.optimization.registry import (
    ParameterOptApplyFunc,
    ParameterOptMatch,
    ParameterOptPattern,
    ParameterOptPatternDefn,
)
from cirkit.backend.torch.parameters.nodes import (
    TorchKroneckerParameter,
    TorchLogParameter,
    TorchLogSoftmaxParameter,
    TorchOuterSumParameter,
    TorchParameterNode,
    TorchReduceLSEParameter,
    TorchSoftmaxParameter,
)

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


class ReduceLSEOuterSumPattern(ParameterOptPatternDefn):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> list[type[TorchParameterNode]]:
        return [TorchReduceLSEParameter, TorchOuterSumParameter]


def apply_log_softmax(
    compiler: "TorchCompiler", match: ParameterOptMatch
) -> tuple[TorchLogSoftmaxParameter]:
    softmax = cast(TorchSoftmaxParameter, match.entries[1])
    log_softmax = TorchLogSoftmaxParameter(softmax.in_shapes[0], dim=softmax.dim)
    return (log_softmax,)


DEFAULT_PARAMETER_OPT_RULES: dict[
    ParameterOptPattern, ParameterOptApplyFunc
] = {  # type: ignore[misc]
    LogSoftmaxPattern: apply_log_softmax,
    # TODO: implement this neat optimization with einsums on the lse-sum semiring
    # ReduceLSEOuterSumPattern: apply_lse_outer_sum_einsum
}
