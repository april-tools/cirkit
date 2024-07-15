from typing import TYPE_CHECKING, Dict, List, Tuple, Type, cast

from cirkit.backend.torch.layers import (
    TorchDenseLayer,
    TorchHadamardLayer,
    TorchKroneckerLayer,
    TorchLayer,
    TorchTuckerLayer,
)
from cirkit.backend.torch.layers.optimized import TorchTensorDotLayer
from cirkit.backend.torch.layers.sum_product import TorchCPLayer
from cirkit.backend.torch.optimization.parameters import KroneckerOutParameterPattern
from cirkit.backend.torch.optimization.registry import (
    LayerOptApplyFunc,
    LayerOptMatch,
    LayerOptPattern,
    ParameterOptPattern,
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


class TuckerPattern(LayerOptPattern):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> List[Type[TorchLayer]]:
        return [TorchDenseLayer, TorchKroneckerLayer]

    @classmethod
    def ppatterns(cls) -> List[Dict[str, ParameterOptPattern]]:
        return [{} for _ in cls.entries()]


class CandecompPattern(LayerOptPattern):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> List[Type[TorchLayer]]:
        return [TorchDenseLayer, TorchHadamardLayer]

    @classmethod
    def ppatterns(cls) -> List[Dict[str, ParameterOptPattern]]:
        return [{} for _ in cls.entries()]


class DenseKroneckerPattern(LayerOptPattern):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> List[Type[TorchLayer]]:
        return [TorchDenseLayer]

    @classmethod
    def ppatterns(cls) -> List[Dict[str, ParameterOptPattern]]:
        return [{"weight": KroneckerOutParameterPattern}]


def apply_tucker(compiler: "TorchCompiler", match: LayerOptMatch) -> Tuple[TorchTuckerLayer]:
    dense = cast(TorchDenseLayer, match.entries[0])
    kronecker = cast(TorchKroneckerLayer, match.entries[1])
    tucker = TorchTuckerLayer(
        kronecker.num_input_units,
        dense.num_output_units,
        kronecker.arity,
        weight=dense.weight,
        semiring=compiler.semiring,
    )
    return (tucker,)


def apply_candecomp(compiler: "TorchCompiler", match: LayerOptMatch) -> Tuple[TorchCPLayer]:
    dense = cast(TorchDenseLayer, match.entries[0])
    hadamard = cast(TorchHadamardLayer, match.entries[1])
    cp = TorchCPLayer(
        hadamard.num_input_units,
        dense.num_output_units,
        hadamard.arity,
        weight=dense.weight,
        semiring=compiler.semiring,
    )
    return (cp,)


def apply_tensordot(
    compiler: "TorchCompiler", match: LayerOptMatch
) -> Tuple[TorchTensorDotLayer, TorchTensorDotLayer]:
    # Retrieve the matched dense layer and the inputs to the kronecker parameter node
    dense = cast(TorchDenseLayer, match.entries[0])
    weight_patterns = match.pentries[0]["weight"]
    kronecker = weight_patterns[0].entries[0]
    weight1_output, weight2_output = dense.weight.node_inputs(kronecker)

    # Build new torch parameter computational graphs by taking
    # the sub-computational graph rooted at the inputs of the kronecker parameter node
    weight1, weight2 = dense.weight.extract_subgraphs(weight1_output, weight2_output)

    # Instantiate two tensor dot layers, whose composition is equivalent to the
    # dense layer parameterized by a kronecker product of two matrices
    tdot1 = TorchTensorDotLayer(
        dense.num_input_units,
        weight1.shape[0] * weight2.shape[1],
        weight2.shape[1],
        weight=weight1,
        semiring=compiler.semiring,
    )
    tdot2 = TorchTensorDotLayer(
        weight1.shape[0] * weight2.shape[1],
        dense.num_output_units,
        weight1.shape[0],
        weight=weight2,
        semiring=compiler.semiring,
    )
    return tdot1, tdot2


DEFAULT_LAYER_FUSE_OPT_RULES: Dict[LayerOptPattern, LayerOptApplyFunc] = {  # type: ignore[misc]
    TuckerPattern: apply_tucker,
    CandecompPattern: apply_candecomp,
}
DEFAULT_LAYER_SHATTER_OPT_RULES: Dict[LayerOptPattern, LayerOptApplyFunc] = {  # type: ignore[misc]
    DenseKroneckerPattern: apply_tensordot
}
