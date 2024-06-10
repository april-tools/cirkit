from typing import TYPE_CHECKING, Dict

from cirkit.backend.base import LayerCompilationFunc, LayerCompilationSign
from cirkit.backend.torch.layers import TorchCategoricalLayer, TorchLogPartitionLayer
from cirkit.backend.torch.layers.inner import (
    TorchDenseLayer,
    TorchHadamardLayer,
    TorchKroneckerLayer,
    TorchMixingLayer,
)
from cirkit.symbolic.layers import (
    CategoricalLayer,
    DenseLayer,
    HadamardLayer,
    KroneckerLayer,
    LogPartitionLayer,
    MixingLayer,
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


def compile_log_partition_layer(
    compiler: "TorchCompiler", sl: LogPartitionLayer
) -> TorchLogPartitionLayer:
    value = compiler.compile_parameter(sl.value)
    return TorchLogPartitionLayer(
        sl.scope,
        sl.num_output_units,
        num_channels=sl.num_channels,
        value=value,
        semiring=compiler.semiring,
    )


def compile_categorical_layer(
    compiler: "TorchCompiler", sl: CategoricalLayer
) -> TorchCategoricalLayer:
    logits = compiler.compile_parameter(sl.logits)
    return TorchCategoricalLayer(
        sl.scope,
        sl.num_output_units,
        num_channels=sl.num_channels,
        num_categories=sl.num_categories,
        logits=logits,
        semiring=compiler.semiring,
    )


def compile_hadamard_layer(compiler: "TorchCompiler", sl: KroneckerLayer) -> TorchHadamardLayer:
    return TorchHadamardLayer(
        sl.num_input_units, sl.num_output_units, arity=sl.arity, semiring=compiler.semiring
    )


def compile_kronecker_layer(compiler: "TorchCompiler", sl: KroneckerLayer) -> TorchKroneckerLayer:
    return TorchKroneckerLayer(
        sl.num_input_units, sl.num_output_units, arity=sl.arity, semiring=compiler.semiring
    )


def compile_dense_layer(compiler: "TorchCompiler", sl: DenseLayer) -> TorchDenseLayer:
    weight = compiler.compile_parameter(sl.weight)
    return TorchDenseLayer(
        sl.num_input_units, sl.num_output_units, weight=weight, semiring=compiler.semiring
    )


def compile_mixing_layer(compiler: "TorchCompiler", sl: MixingLayer) -> TorchMixingLayer:
    weight = compiler.compile_parameter(sl.weight)
    return TorchMixingLayer(
        sl.num_input_units,
        sl.num_output_units,
        arity=sl.arity,
        weight=weight,
        semiring=compiler.semiring,
    )


DEFAULT_LAYER_COMPILATION_RULES: Dict[LayerCompilationSign, LayerCompilationFunc] = {  # type: ignore[misc]
    LogPartitionLayer: compile_log_partition_layer,
    CategoricalLayer: compile_categorical_layer,
    HadamardLayer: compile_hadamard_layer,
    KroneckerLayer: compile_kronecker_layer,
    DenseLayer: compile_dense_layer,
    MixingLayer: compile_mixing_layer,
}
