import torch

from cirkit.backend.torch.compiler import TorchCompiler
from tests.symbolic.test_utils import build_circuit


def test_compile_simple_circuit():
    compiler = TorchCompiler()
    num_variables, num_channels = 12, 1
    sc = build_circuit(num_variables, 4, 2, num_repetitions=3)
    tc = compiler.compile(sc)
    batch_size = 32
    input_shape = (batch_size, num_channels, num_variables)
    x = torch.zeros(input_shape)
    y = tc(x)
    assert y.shape == (batch_size, 1, 1)
