from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.layers import (
    TorchCategoricalLayer,
    TorchConstantLayer,
    TorchGaussianLayer,
)
from cirkit.backend.torch.layers.inner import (
    TorchDenseLayer,
    TorchHadamardLayer,
    TorchKroneckerLayer,
    TorchMixingLayer,
)
from cirkit.backend.torch.params import TorchParameter
from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.params.composed import TorchKroneckerParameter
from cirkit.symbolic.layers import (
    CategoricalLayer,
    ConstantLayer,
    DenseLayer,
    GaussianLayer,
    KroneckerLayer,
    MixingLayer,
    PlaceholderParameter,
)
from cirkit.symbolic.params import OuterProductParameter, Parameter


def compile_parameter(parameter: Parameter, compiler: TorchCompiler) -> TorchParameter:
    return TorchParameter(*parameter.shape)


def compile_placeholder_parameter(
    parameter: PlaceholderParameter, compiler: TorchCompiler
) -> AbstractTorchParameter:
    return compiler.retrieve_parameter(parameter.layer, parameter.name)


def compile_kronecker_parameter(
    parameter: OuterProductParameter, compiler: TorchCompiler
) -> TorchKroneckerParameter:
    return TorchKroneckerParameter(
        compiler.compile_parameter(parameter.p1),
        compiler.compile_parameter(parameter.p2),
        dim=parameter.axis,
    )


def compile_constant_layer(sl: ConstantLayer, compiler: TorchCompiler) -> TorchConstantLayer:
    return TorchConstantLayer(
        num_input_units=sl.num_input_units,
        num_output_units=sl.num_output_units,
        arity=sl.arity,
        value=compiler.compile_parameter(sl.value),
    )


def compile_categorical_layer(
    sl: CategoricalLayer, compiler: TorchCompiler
) -> TorchCategoricalLayer:
    return TorchCategoricalLayer(
        num_input_units=sl.num_input_units,
        num_output_units=sl.num_output_units,
        arity=sl.num_channels,
        num_categories=sl.num_categories,
        param=compiler.compile_parameter(sl.probs),
    )


def compile_gaussian_layer(sl: GaussianLayer, compiler: TorchCompiler) -> TorchGaussianLayer:
    return TorchGaussianLayer(
        num_input_units=sl.num_input_units,
        num_output_units=sl.num_output_units,
        arity=sl.num_channels,
        mean=compiler.compile_parameter(sl.mean),
        stddev=compiler.compile_parameter(sl.stddev),
    )


def compile_hadamard_layer(sl: KroneckerLayer, compiler: TorchCompiler) -> TorchHadamardLayer:
    return TorchHadamardLayer(
        num_input_units=sl.num_input_units, num_output_units=sl.num_output_units, arity=sl.arity
    )


def compile_kronecker_layer(sl: KroneckerLayer, compiler: TorchCompiler) -> TorchKroneckerLayer:
    return TorchKroneckerLayer(
        num_input_units=sl.num_input_units, num_output_units=sl.num_output_units, arity=sl.arity
    )


def compile_dense_layer(sl: DenseLayer, compiler: TorchCompiler) -> TorchDenseLayer:
    return TorchDenseLayer(
        num_input_units=sl.num_input_units,
        num_output_units=sl.num_output_units,
        param=compiler.compile_parameter(sl.weight),
    )


def compile_mixing_layer(sl: MixingLayer, compiler: TorchCompiler) -> TorchMixingLayer:
    return TorchMixingLayer(
        num_input_units=sl.num_input_units,
        num_output_units=sl.num_output_units,
        arity=sl.arity,
        param=compiler.compile_parameter(sl.weight),
    )
