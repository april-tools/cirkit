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
from cirkit.backend.torch.parameters.nodes import TorchKroneckerParameter
from cirkit.backend.torch.parameters.parameter import TorchParameter

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


class TensorDotKroneckerPattern(LayerOptPattern):
    @classmethod
    def is_output(cls) -> bool:
        return False

    @classmethod
    def entries(cls) -> List[Type[TorchLayer]]:
        return [TorchTensorDotLayer]

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


def _apply_tensordot_rule(
    compiler: "TorchCompiler",
    num_input_units: int,
    num_output_units: int,
    weight: TorchParameter,
    kronecker: TorchKroneckerParameter,
) -> Tuple[TorchTensorDotLayer, TorchTensorDotLayer]:
    # Build new torch parameter computational graphs by taking
    # the sub-computational graph rooted at the inputs of the kronecker parameter node
    weight1, weight2 = weight.extract_subgraphs(*weight.node_inputs(kronecker))

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
) -> Tuple[TorchTensorDotLayer, TorchTensorDotLayer]:
    dense = cast(TorchDenseLayer, match.entries[0])
    weight_patterns = match.pentries[0]["weight"]
    kronecker = cast(TorchKroneckerParameter, weight_patterns[0].entries[0])
    return _apply_tensordot_rule(
        compiler, dense.num_input_units, dense.num_output_units, dense.weight, kronecker
    )


def apply_tensordot_tensordot(
    compiler: "TorchCompiler", match: LayerOptMatch
) -> Tuple[TorchTensorDotLayer, TorchTensorDotLayer]:
    tdot = cast(TorchTensorDotLayer, match.entries[0])
    weight_patterns = match.pentries[0]["weight"]
    kronecker = cast(TorchKroneckerParameter, weight_patterns[0].entries[0])
    return _apply_tensordot_rule(
        compiler, tdot.num_input_units, tdot.num_output_units, tdot.weight, kronecker
    )


DEFAULT_LAYER_FUSE_OPT_RULES: Dict[LayerOptPattern, LayerOptApplyFunc] = {  # type: ignore[misc]
    TuckerPattern: apply_tucker,
    CandecompPattern: apply_candecomp,
}
DEFAULT_LAYER_SHATTER_OPT_RULES: Dict[LayerOptPattern, LayerOptApplyFunc] = {  # type: ignore[misc]
    DenseKroneckerPattern: apply_dense_tensordot,
    TensorDotKroneckerPattern: apply_tensordot_tensordot,
}
