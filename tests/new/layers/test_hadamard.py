import torch

from cirkit.new.layers.inner.product.hadamard import HadamardLayer


def test_hadamard_layer1() -> None:
    hadamard_layer = HadamardLayer(num_input_units=2, num_output_units=2, arity=2, reparam=None)
    input_tensor = torch.rand((3, 3, 4))
    output_tensor = hadamard_layer.forward(input_tensor)

    assert output_tensor.shape == torch.Size([3, 4])


def test_hadamard_layer2() -> None:
    hadamard_layer = HadamardLayer(num_input_units=3, num_output_units=3, arity=2, reparam=None)
    input_tensor = torch.rand((3, 4, 5))
    output_tensor = hadamard_layer.forward(input_tensor)

    assert output_tensor.shape == torch.Size([4, 5])
