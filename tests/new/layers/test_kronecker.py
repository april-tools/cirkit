import torch

from cirkit.new.layers.inner.product.kronecker import KroneckerLayer


def test_kronecker_layer1() -> None:
    k_layer = KroneckerLayer(num_input_units=2, num_output_units=4, arity=2, reparam=None)
    input_tensor = torch.rand((3, 3, 4))
    output_tensor = k_layer.forward(input_tensor)

    assert output_tensor.shape == torch.Size([3, 16])


def test_hadamard_layer2() -> None:
    hadamard_layer = KroneckerLayer(num_input_units=3, num_output_units=9, arity=2, reparam=None)
    input_tensor = torch.rand((3, 4, 5))
    output_tensor = hadamard_layer.forward(input_tensor)

    assert output_tensor.shape == torch.Size([4, 25])
