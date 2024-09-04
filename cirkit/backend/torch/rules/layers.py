from typing import TYPE_CHECKING, Dict

import torch

from cirkit.backend.compiler import LayerCompilationFunc, LayerCompilationSign
from cirkit.backend.torch.layers.inner import (
    TorchDenseLayer,
    TorchHadamardLayer,
    TorchKroneckerLayer,
    TorchMixingLayer,
)
from cirkit.backend.torch.layers.input import (
    TorchCategoricalLayer,
    TorchGaussianLayer,
    TorchLogPartitionLayer,
    TorchPolynomialLayer,
)
from cirkit.symbolic.layers import (
    CategoricalLayer,
    DenseLayer,
    GaussianLayer,
    HadamardLayer,
    KroneckerLayer,
    LogPartitionLayer,
    MixingLayer,
    PolynomialLayer,
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


def compile_log_partition_layer(
    compiler: "TorchCompiler", sl: LogPartitionLayer
) -> TorchLogPartitionLayer:
    value = compiler.compile_parameter(sl.value)
    return TorchLogPartitionLayer(
        torch.tensor(tuple(sl.scope)),
        sl.num_output_units,
        num_channels=sl.num_channels,
        value=value,
        semiring=compiler.semiring,
    )


def compile_categorical_layer(
    compiler: "TorchCompiler", sl: CategoricalLayer
) -> TorchCategoricalLayer:
    if sl.logits is None:
        probs = compiler.compile_parameter(sl.probs)
        logits = None
    else:
        probs = None
        logits = compiler.compile_parameter(sl.logits)
    return TorchCategoricalLayer(
        torch.tensor(tuple(sl.scope)),
        sl.num_output_units,
        num_channels=sl.num_channels,
        num_categories=sl.num_categories,
        probs=probs,
        logits=logits,
        semiring=compiler.semiring,
    )


def compile_gaussian_layer(compiler: "TorchCompiler", sl: GaussianLayer) -> TorchGaussianLayer:
    mean = compiler.compile_parameter(sl.mean)
    stddev = compiler.compile_parameter(sl.stddev)
    if sl.log_partition is not None:
        log_partition = compiler.compile_parameter(sl.log_partition)
    else:
        log_partition = None
    return TorchGaussianLayer(
        torch.tensor(tuple(sl.scope)),
        sl.num_output_units,
        num_channels=sl.num_channels,
        mean=mean,
        stddev=stddev,
        log_partition=log_partition,
        semiring=compiler.semiring,
    )


def compile_polynomial_layer(
    compiler: "TorchCompiler", sl: PolynomialLayer
) -> TorchPolynomialLayer:
    coeff = compiler.compile_parameter(sl.coeff)
    return TorchPolynomialLayer(
        torch.tensor(tuple(sl.scope)),
        sl.num_output_units,
        num_channels=sl.num_channels,
        degree=sl.degree,
        coeff=coeff,
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
    GaussianLayer: compile_gaussian_layer,
    PolynomialLayer: compile_polynomial_layer,
    HadamardLayer: compile_hadamard_layer,
    KroneckerLayer: compile_kronecker_layer,
    DenseLayer: compile_dense_layer,
    MixingLayer: compile_mixing_layer,
}
