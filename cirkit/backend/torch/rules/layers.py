from typing import TYPE_CHECKING, cast

import torch

from cirkit.backend.compiler import LayerCompilationFunc, LayerCompilationSign
from cirkit.backend.torch.layers.inner import TorchHadamardLayer, TorchKroneckerLayer, TorchSumLayer
from cirkit.backend.torch.layers.input import (
    TorchBinomialLayer,
    TorchCategoricalLayer,
    TorchConstantValueLayer,
    TorchEmbeddingLayer,
    TorchEvidenceLayer,
    TorchGaussianLayer,
    TorchInputLayer,
    TorchPolynomialLayer,
)
from cirkit.symbolic.layers import (
    BinomialLayer,
    CategoricalLayer,
    ConstantValueLayer,
    EmbeddingLayer,
    EvidenceLayer,
    GaussianLayer,
    HadamardLayer,
    KroneckerLayer,
    PolynomialLayer,
    SumLayer,
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


def compile_embedding_layer(compiler: "TorchCompiler", sl: EmbeddingLayer) -> TorchEmbeddingLayer:
    weight = compiler.compile_parameter(sl.weight)
    return TorchEmbeddingLayer(
        torch.tensor(tuple(sl.scope)),
        sl.num_output_units,
        num_channels=sl.num_channels,
        num_states=sl.num_states,
        weight=weight,
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


def compile_binomial_layer(compiler: "TorchCompiler", sl: BinomialLayer) -> TorchBinomialLayer:
    if sl.logits is None:
        probs = compiler.compile_parameter(sl.probs)
        logits = None
    else:
        probs = None
        logits = compiler.compile_parameter(sl.logits)
    return TorchBinomialLayer(
        torch.tensor(tuple(sl.scope)),
        sl.num_output_units,
        num_channels=sl.num_channels,
        total_count=sl.total_count,
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
    return TorchHadamardLayer(sl.num_input_units, arity=sl.arity, semiring=compiler.semiring)


def compile_kronecker_layer(compiler: "TorchCompiler", sl: KroneckerLayer) -> TorchKroneckerLayer:
    return TorchKroneckerLayer(sl.num_input_units, arity=sl.arity, semiring=compiler.semiring)


def compile_sum_layer(compiler: "TorchCompiler", sl: SumLayer) -> TorchSumLayer:
    weight = compiler.compile_parameter(sl.weight)
    return TorchSumLayer(
        sl.num_input_units,
        sl.num_output_units,
        arity=sl.arity,
        weight=weight,
        semiring=compiler.semiring,
    )


def compile_constant_value_layer(
    compiler: "TorchCompiler", sl: ConstantValueLayer
) -> TorchConstantValueLayer:
    value = compiler.compile_parameter(sl.value)
    return TorchConstantValueLayer(
        sl.num_output_units,
        log_space=sl.log_space,
        value=value,
        semiring=compiler.semiring,
    )


def compile_evidence_layer(compiler: "TorchCompiler", sl: EvidenceLayer) -> TorchEvidenceLayer:
    layer = compiler.compile_layer(sl.layer)
    observation = compiler.compile_parameter(sl.observation)
    return TorchEvidenceLayer(
        cast(TorchInputLayer, layer), observation=observation, semiring=compiler.semiring
    )


DEFAULT_LAYER_COMPILATION_RULES: dict[LayerCompilationSign, LayerCompilationFunc] = {  # type: ignore[misc]
    EmbeddingLayer: compile_embedding_layer,
    CategoricalLayer: compile_categorical_layer,
    BinomialLayer: compile_binomial_layer,
    GaussianLayer: compile_gaussian_layer,
    PolynomialLayer: compile_polynomial_layer,
    HadamardLayer: compile_hadamard_layer,
    KroneckerLayer: compile_kronecker_layer,
    SumLayer: compile_sum_layer,
    ConstantValueLayer: compile_constant_value_layer,
    EvidenceLayer: compile_evidence_layer,
}
