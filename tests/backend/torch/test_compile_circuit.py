import itertools

import pytest
import torch

import cirkit.symbolic.functional as SF
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.models import TensorizedCircuit, TensorizedConstantCircuit
from tests.floats import allclose, isclose
from tests.symbolic.test_utils import build_simple_circuit, build_simple_pc


def test_compile_output_shape():
    compiler = TorchCompiler()
    num_variables, num_channels = 12, 1
    sc = build_simple_circuit(num_variables, 4, 3, num_repetitions=3)
    tc: TensorizedCircuit = compiler.compile(sc)
    batch_size = 32
    input_shape = (batch_size, num_channels, num_variables)
    x = torch.zeros(input_shape)
    y = tc(x)
    assert y.shape == (batch_size, 1, 1)
    assert torch.all(torch.isfinite(y))


@pytest.mark.parametrize("normalized", [True])
def test_compile_integrate_pc(normalized: bool):
    compiler = TorchCompiler()
    num_variables, num_channels = 7, 1
    sc = build_simple_pc(num_variables, 4, 3, num_repetitions=3, normalized=normalized)
    int_sc = SF.integrate(sc)
    int_tc: TensorizedConstantCircuit = compiler.compile(int_sc)
    tc: TensorizedCircuit = compiler.get_compiled_circuit(sc)
    assert isinstance(tc, TensorizedCircuit) and isinstance(int_tc, TensorizedConstantCircuit)

    z = int_tc()  # compute the partition function
    assert z.shape == (1, 1)
    z = z.squeeze()
    if normalized:
        assert isclose(z.item(), 1.0)
    else:
        assert not isclose(z.item(), 1.0)

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2 ** num_variables, 1, num_variables)
    scores = tc(worlds)
    assert scores.shape == (2 ** num_variables, 1, 1)
    scores = scores.squeeze()
    if normalized:
        assert isclose(torch.sum(scores), 1.0)
    else:
        assert isclose(torch.sum(scores), z)
