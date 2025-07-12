# pylint: disable=bad-mcs-classmethod-argument

from typing import TYPE_CHECKING, Any, cast

from cirkit.backend.torch.layers import (
    TorchHadamardLayer,
    TorchKroneckerLayer,
    TorchLayer,
    TorchSumLayer,
    TorchTuckerLayer,
)
from cirkit.backend.torch.layers.optimized import TorchCPTLayer, TorchTensorDotLayer
from cirkit.backend.torch.optimization.parameters import KroneckerOutParameterPattern
from cirkit.backend.torch.optimization.registry import (
    LayerOptApplyFunc,
    LayerOptMatch,
    LayerOptPattern,
    ParameterOptPattern,
)
from cirkit.backend.torch.parameters.nodes import TorchKroneckerParameter, TorchMatMulParameter
from cirkit.backend.torch.parameters.parameter import TorchParameter

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


class SumCollapsePattern(LayerOptPattern):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> list[type[TorchLayer]]:
        return [TorchSumLayer, TorchSumLayer]

    @classmethod
    def ppatterns(cls) -> list[dict[str, ParameterOptPattern]]:
        return [{} for _ in cls.entries()]

    @classmethod
    def cpatterns(cls) -> list[dict[str, Any]]:
        return [{"arity": 1}, {}]


class TuckerPattern(LayerOptPattern):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> list[type[TorchLayer]]:
        return [TorchSumLayer, TorchKroneckerLayer]

    @classmethod
    def ppatterns(cls) -> list[dict[str, ParameterOptPattern]]:
        return [{} for _ in cls.entries()]

    @classmethod
    def cpatterns(cls) -> list[dict[str, Any]]:
        return [{"arity": 1}, {}]


class CandecompPattern(LayerOptPattern):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> list[type[TorchLayer]]:
        return [TorchSumLayer, TorchHadamardLayer]

    @classmethod
    def ppatterns(cls) -> list[dict[str, ParameterOptPattern]]:
        return [{} for _ in cls.entries()]

    @classmethod
    def cpatterns(cls) -> list[dict[str, Any]]:
        return [{"arity": 1}, {}]


class DenseKroneckerPattern(LayerOptPattern):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> list[type[TorchLayer]]:
        return [TorchSumLayer]

    @classmethod
    def ppatterns(cls) -> list[dict[str, ParameterOptPattern]]:
        return [{"weight": KroneckerOutParameterPattern}]

    @classmethod
    def cpatterns(cls) -> list[dict[str, Any]]:
        return [{"arity": 1}]


class TensorDotKroneckerPattern(LayerOptPattern):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> list[type[TorchLayer]]:
        return [TorchTensorDotLayer]

    @classmethod
    def ppatterns(cls) -> list[dict[str, ParameterOptPattern]]:
        return [{"weight": KroneckerOutParameterPattern}]

    @classmethod
    def cpatterns(cls) -> list[dict[str, Any]]:
        return [{}]


def apply_sum_collapse(compiler: "TorchCompiler", match: LayerOptMatch) -> tuple[TorchSumLayer]:
    dense1 = cast(TorchSumLayer, match.entries[0])
    dense2 = cast(TorchSumLayer, match.entries[1])
    weight = TorchParameter.from_binary(
        TorchMatMulParameter(dense1.weight.shape, dense2.weight.shape), dense1.weight, dense2.weight
    )
    dense = TorchSumLayer(
        dense2.num_input_units,
        dense1.num_output_units,
        arity=dense2.arity,
        weight=weight,
        semiring=compiler.semiring,
    )
    return (dense,)


def apply_tucker(compiler: "TorchCompiler", match: LayerOptMatch) -> tuple[TorchTuckerLayer]:
    dense = cast(TorchSumLayer, match.entries[0])
    kronecker = cast(TorchKroneckerLayer, match.entries[1])
    tucker = TorchTuckerLayer(
        kronecker.num_input_units,
        dense.num_output_units,
        kronecker.arity,
        weight=dense.weight,
        semiring=compiler.semiring,
    )
    return (tucker,)


def apply_candecomp(compiler: "TorchCompiler", match: LayerOptMatch) -> tuple[TorchCPTLayer]:
    dense = cast(TorchSumLayer, match.entries[0])
    hadamard = cast(TorchHadamardLayer, match.entries[1])
    cpt = TorchCPTLayer(
        hadamard.num_input_units,
        dense.num_output_units,
        hadamard.arity,
        weight=dense.weight,
        semiring=compiler.semiring,
    )
    return (cpt,)


def _apply_tensordot_rule(
    compiler: "TorchCompiler",
    num_input_units: int,
    num_output_units: int,
    weight: TorchParameter,
    kronecker: TorchKroneckerParameter,
) -> tuple[TorchTensorDotLayer, TorchTensorDotLayer]:
    # Build new torch parameter computational graphs by taking
    # the sub-computational graph rooted at the inputs of the kronecker parameter node
    in_kronecker1, in_kronecker2 = weight.node_inputs(kronecker)
    weight1 = weight.subgraph(in_kronecker1)
    weight2 = weight.subgraph(in_kronecker2)

    # Instantiate two tensor dot layers
    num_inner_units = weight1.shape[0] * (num_input_units // weight1.shape[1])
    tdot1 = TorchTensorDotLayer(
        num_input_units,
        num_inner_units,
        weight=weight1,
        semiring=compiler.semiring,
    )
    tdot2 = TorchTensorDotLayer(
        num_inner_units,
        num_output_units,
        weight=weight2,
        semiring=compiler.semiring,
    )
    return tdot1, tdot2


def apply_dense_tensordot(
    compiler: "TorchCompiler", match: LayerOptMatch
) -> tuple[TorchTensorDotLayer, TorchTensorDotLayer]:
    dense = cast(TorchSumLayer, match.entries[0])
    weight_patterns = match.pentries[0]["weight"]
    kronecker = cast(TorchKroneckerParameter, weight_patterns[0].entries[0])
    return _apply_tensordot_rule(
        compiler, dense.num_input_units, dense.num_output_units, dense.weight, kronecker
    )


def apply_tensordot_tensordot(
    compiler: "TorchCompiler", match: LayerOptMatch
) -> tuple[TorchTensorDotLayer, TorchTensorDotLayer]:
    tdot = cast(TorchTensorDotLayer, match.entries[0])
    weight_patterns = match.pentries[0]["weight"]
    kronecker = cast(TorchKroneckerParameter, weight_patterns[0].entries[0])
    return _apply_tensordot_rule(
        compiler, tdot.num_input_units, tdot.num_output_units, tdot.weight, kronecker
    )


DEFAULT_LAYER_FUSE_OPT_RULES: dict[LayerOptPattern, LayerOptApplyFunc] = {  # type: ignore[misc]
    SumCollapsePattern: apply_sum_collapse,
    TuckerPattern: apply_tucker,
    CandecompPattern: apply_candecomp,
}
DEFAULT_LAYER_SHATTER_OPT_RULES: dict[LayerOptPattern, LayerOptApplyFunc] = {  # type: ignore[misc]
    DenseKroneckerPattern: apply_dense_tensordot,
    TensorDotKroneckerPattern: apply_tensordot_tensordot,
}
