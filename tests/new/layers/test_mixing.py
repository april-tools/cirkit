import torch

from cirkit.new.layers.inner.sum.mixing import MixingLayer
from cirkit.new.reparams.unary import UnaryReparam


def test_mixing_layer() -> None:
    layer = MixingLayer(
        num_input_units=2, num_output_units=2, arity=2, reparam=UnaryReparam(func=torch.tanh)
    )
    input_tensor = torch.rand((2, 2))
    output_tensor = layer.forward(input_tensor)
    assert output_tensor.shape == torch.Size([2])
