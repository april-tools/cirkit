from typing import TYPE_CHECKING, Dict

from cirkit.backend.base import (
    LayerCompilationFunc,
    LayerCompilationSign,
    ParameterCompilationFunc,
    ParameterCompilationSign,
)
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
from cirkit.backend.torch.utils import InitializerFunc
from cirkit.symbolic.layers import (
    CategoricalLayer,
    ConstantLayer,
    DenseLayer,
    GaussianLayer,
    HadamardLayer,
    KroneckerLayer,
    MixingLayer,
    PlaceholderParameter,
)
from cirkit.symbolic.params import ConstantParameter, Parameter

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


def compile_parameter(
    p: Parameter, init_func: InitializerFunc, compiler: "TorchCompiler"
) -> TorchParameter:
    compiled_p = TorchParameter(*p.shape)
    compiled_p.initialize(init_func)
    return compiled_p


def compile_placeholder_parameter(
    p: PlaceholderParameter, init_func: InitializerFunc, compiler: "TorchCompiler"
) -> AbstractTorchParameter:
    return compiler.retrieve_parameter(p.layer, p.name)


def compile_constant_parameter(
    p: ConstantParameter, init_func: InitializerFunc, compiler: "TorchCompiler"
) -> TorchParameter:
    ...


def compile_kronecker_parameter(
    p: TorchKroneckerParameter, init_func: InitializerFunc, compiler: "TorchCompiler"
) -> TorchKroneckerParameter:
    ...


def compile_constant_layer(sl: ConstantLayer, compiler: "TorchCompiler") -> TorchConstantLayer:
    return TorchConstantLayer(
        num_variables=sl.num_variables,
        num_output_units=sl.num_output_units,
        num_channels=sl.num_channels,
        value=compiler.compile_parameter(sl.value),
    )


def compile_categorical_layer(
    sl: CategoricalLayer, compiler: "TorchCompiler"
) -> TorchCategoricalLayer:
    logits = compiler.compile_parameter(
        sl.logits, compiler.retrieve_initializer(TorchCategoricalLayer, "logits")
    )
    return TorchCategoricalLayer(
        sl.num_variables,
        sl.num_output_units,
        num_channels=sl.num_channels,
        num_categories=sl.num_categories,
        logits=logits,
    )


def compile_gaussian_layer(sl: GaussianLayer, compiler: "TorchCompiler") -> TorchGaussianLayer:
    return TorchGaussianLayer(
        num_variables=sl.num_variables,
        num_output_units=sl.num_output_units,
        num_channels=sl.num_channels,
        mean=compiler.compile_parameter(sl.mean),
        stddev=compiler.compile_parameter(sl.stddev),
    )


def compile_hadamard_layer(sl: KroneckerLayer, compiler: "TorchCompiler") -> TorchHadamardLayer:
    return TorchHadamardLayer(
        num_input_units=sl.num_input_units, num_output_units=sl.num_output_units, arity=sl.arity
    )


def compile_kronecker_layer(sl: KroneckerLayer, compiler: "TorchCompiler") -> TorchKroneckerLayer:
    return TorchKroneckerLayer(
        num_input_units=sl.num_input_units, num_output_units=sl.num_output_units, arity=sl.arity
    )


def compile_dense_layer(sl: DenseLayer, compiler: "TorchCompiler") -> TorchDenseLayer:
    weight = compiler.compile_parameter(
        sl.weight, compiler.retrieve_initializer(TorchDenseLayer, "weight")
    )
    return TorchDenseLayer(sl.num_input_units, sl.num_output_units, weight=weight)


def compile_mixing_layer(sl: MixingLayer, compiler: "TorchCompiler") -> TorchMixingLayer:
    weight = compiler.compile_parameter(
        sl.weight, compiler.retrieve_initializer(TorchDenseLayer, "weight")
    )
    return TorchMixingLayer(sl.num_input_units, sl.num_output_units, arity=sl.arity, weight=weight)


DEFAULT_LAYER_COMPILATION_RULES: Dict[LayerCompilationSign, LayerCompilationFunc] = {  # type: ignore[misc]
    CategoricalLayer: compile_categorical_layer,
    ConstantLayer: compile_constant_layer,
    GaussianLayer: compile_gaussian_layer,
    HadamardLayer: compile_hadamard_layer,
    KroneckerLayer: compile_kronecker_layer,
    DenseLayer: compile_dense_layer,
    MixingLayer: compile_mixing_layer,
}
DEFAULT_PARAMETER_COMPILATION_RULES: Dict[ParameterCompilationSign, ParameterCompilationFunc] = {  # type: ignore[misc]
    Parameter: compile_parameter,
    PlaceholderParameter: compile_placeholder_parameter,
}
