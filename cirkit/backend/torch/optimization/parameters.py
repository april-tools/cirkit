from typing import TYPE_CHECKING, Dict, List, Tuple, Type, cast

from cirkit.backend.torch.optimization.registry import (
    ParameterOptApplyFunc,
    ParameterOptMatch,
    ParameterOptPattern,
    ParameterOptPatternDefn,
)
from cirkit.backend.torch.parameters.leaves import TorchParameterNode
from cirkit.backend.torch.parameters.ops import (
    TorchKroneckerParameter,
    TorchLogParameter,
    TorchLogSoftmaxParameter,
    TorchOuterSumParameter,
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
    def entries(cls) -> List[Type[TorchParameterNode]]:
        return [TorchKroneckerParameter]


class LogSoftmaxPattern(ParameterOptPatternDefn):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> List[Type[TorchParameterNode]]:
        return [TorchLogParameter, TorchSoftmaxParameter]


class ReduceLSEOuterSumPattern(ParameterOptPatternDefn):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> List[Type[TorchParameterNode]]:
        return [TorchReduceLSEParameter, TorchOuterSumParameter]


def apply_log_softmax(
    compiler: "TorchCompiler", match: ParameterOptMatch
) -> Tuple[TorchLogSoftmaxParameter]:
    softmax = cast(TorchSoftmaxParameter, match.entries[1])
    log_softmax = TorchLogSoftmaxParameter(softmax.in_shapes[0], dim=softmax.dim)
    return (log_softmax,)


DEFAULT_PARAMETER_OPT_RULES: Dict[
    ParameterOptPattern, ParameterOptApplyFunc
] = {  # type: ignore[misc]
    LogSoftmaxPattern: apply_log_softmax,
    # TODO: implement this neat optimization with einsums on the lse-sum semiring
    # ReduceLSEOuterSumPattern: apply_lse_outer_sum_einsum
}
