from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.layers import DenseLayer, KroneckerLayer
from cirkit.symbolic.layers import DenseLayer, KroneckerLayer


def compile_dense_layer(sl: DenseLayer, compiler: TorchCompiler) -> DenseLayer:
    return DenseLayer(
        num_input_units=sl.num_input_units,
        num_output_units=sl.num_output_units,
        reparam=compiler.compile_learnable_parameter(sl.weight),
    )


def compile_kronecker_layer(sl: KroneckerLayer, compiler: TorchCompiler) -> DenseLayer:
    return KroneckerLayer(
        num_input_units=sl.num_input_units, num_output_units=sl.num_output_units, arity=sl.arity
    )
