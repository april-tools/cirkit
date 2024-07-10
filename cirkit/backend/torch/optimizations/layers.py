from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Tuple, Type, cast

from cirkit.backend.torch.graph.optimize import GraphOptMatch, GraphOptPatternDefn
from cirkit.backend.torch.layers import (
    TorchDenseLayer,
    TorchHadamardLayer,
    TorchKroneckerLayer,
    TorchLayer,
    TorchTuckerLayer,
)
from cirkit.backend.torch.layers.sum_product import TorchCPLayer
from cirkit.backend.torch.optimizations.parameters import ParameterOptMatch, ParameterOptPattern
from cirkit.backend.torch.parameters.ops import TorchKroneckerParameter, TorchMatMulParameter
from cirkit.backend.torch.parameters.parameter import TorchParameter

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


CircuitOptPatternDefn = GraphOptPatternDefn[TorchLayer]

CircuitOptPattern = Type[CircuitOptPatternDefn]

CircuitOptMatch = GraphOptMatch[TorchLayer]


@dataclass(frozen=True)
class LayerOptPatternDefn:
    cls: Type[TorchLayer]
    patterns: Dict[str, ParameterOptPattern]


LayerOptPattern = Type[LayerOptPatternDefn]


@dataclass(frozen=True)
class LayerOptMatch:
    pattern: LayerOptPattern
    entry: TorchLayer
    matches: Dict[str, ParameterOptMatch]


CircuitOptApplyFunc = Callable[["TorchCompiler", CircuitOptMatch], TorchLayer]
LayerOptApplyFunc = Callable[["TorchCompiler", LayerOptMatch], Tuple[TorchLayer]]


class DenseCompositionPattern(CircuitOptPatternDefn):
    entries = {
        0: TorchDenseLayer,
        1: TorchDenseLayer,
    }


class TuckerPattern(CircuitOptPatternDefn):
    entries = {
        0: TorchDenseLayer,
        1: TorchKroneckerLayer,
    }


class CandecompPattern(CircuitOptPatternDefn):
    entries = {
        0: TorchDenseLayer,
        1: TorchHadamardLayer,
    }


class KroneckerOutputPattern(ParameterOptPattern):
    entries = {0: TorchKroneckerParameter}
    output = True


class DenseKroneckerPattern(LayerOptPatternDefn):
    cls = TorchDenseLayer
    patterns = {"weight": KroneckerOutputPattern}


def apply_dense_composition(compiler: "TorchCompiler", match: CircuitOptMatch) -> TorchDenseLayer:
    dense1 = cast(TorchDenseLayer, match.entries[0])
    dense2 = cast(TorchDenseLayer, match.entries[1])
    weight = TorchParameter.from_binary(
        dense1.weight, dense2.weight, TorchMatMulParameter(dense1.weight.shape, dense2.weight.shape)
    )
    return TorchDenseLayer(
        dense2.num_input_units, dense1.num_output_units, weight=weight, semiring=compiler.semiring
    )


def apply_tucker(compiler: "TorchCompiler", match: CircuitOptMatch) -> TorchTuckerLayer:
    dense = cast(TorchDenseLayer, match.entries[0])
    kronecker = cast(TorchKroneckerLayer, match.entries[1])
    return TorchTuckerLayer(
        kronecker.num_input_units,
        dense.num_output_units,
        kronecker.arity,
        weight=dense.weight,
        semiring=compiler.semiring,
    )


def apply_candecomp(compiler: "TorchCompiler", match: CircuitOptMatch) -> TorchCPLayer:
    dense = cast(TorchDenseLayer, match.entries[0])
    hadamard = cast(TorchHadamardLayer, match.entries[1])
    return TorchCPLayer(
        hadamard.num_input_units,
        dense.num_output_units,
        hadamard.arity,
        weight=dense.weight,
        semiring=compiler.semiring,
    )


def apply_dense_kronecker(
    compiler: "TorchCompiler", match: LayerOptMatch
) -> Tuple[TorchLayer, ...]:
    dense = cast(TorchDenseLayer, match.entry)
    weight = dense.weight
    kronecker = match.matches["weight"].entries[0]
    weight1_output, weight2_output = weight.node_inputs(kronecker)
    # Take the sub-graphs from 'weight' rooted in 'weight1_output' and 'weight2_output',
    # then copy them, and assign them to two distinct dense layers
    raise NotImplementedError()


DEFAULT_CIRCUIT_OPT_RULES: Dict[CircuitOptPattern, CircuitOptApplyFunc] = {  # type: ignore[misc]
    DenseCompositionPattern: apply_dense_composition,
    TuckerPattern: apply_tucker,
    CandecompPattern: apply_candecomp,
}
DEFAULT_LAYER_OPT_RULES: Dict[LayerOptPattern, LayerOptApplyFunc] = {  # type: ignore[misc]
    DenseKroneckerPattern: apply_dense_kronecker
}
