from cirkit.new.layers.inner.sum.dense import DenseLayer
from cirkit.new.reparams.unary import UnaryReparam
from cirkit.new.reparams.binary import BinaryReparam
import torch


def test_dense_layer() -> None:
    layer = DenseLayer(
        num_input_units=3, num_output_units=3, arity=1, reparam=UnaryReparam(func=torch.tanh)
    )
    input_tensor = torch.rand((3))
    output_tensor = layer.forward(input_tensor)
    assert output_tensor.shape == torch.Size([3])
