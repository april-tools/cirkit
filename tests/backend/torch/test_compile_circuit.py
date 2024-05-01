import itertools
from typing import List

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
    batch_size = 42
    input_shape = (batch_size, num_channels, num_variables)
    x = torch.zeros(input_shape)
    y = tc(x)
    assert y.shape == (batch_size, 1, 1)
    assert torch.all(torch.isfinite(y))


@pytest.mark.parametrize("normalized", [False, True])
def test_compile_integrate_pc(normalized: bool):
    compiler = TorchCompiler()
    num_variables, num_channels = 5, 1
    sc = build_simple_pc(num_variables, 4, 3, num_repetitions=3, normalized=normalized)
    int_sc = SF.integrate(sc)
    int_tc: TensorizedConstantCircuit = compiler.compile(int_sc)
    assert isinstance(int_tc, TensorizedConstantCircuit)
    tc: TensorizedCircuit = compiler.get_compiled_circuit(sc)
    assert isinstance(tc, TensorizedCircuit)

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


@pytest.mark.parametrize("normalized,num_products", itertools.product([False, True], [2]))
def test_compile_product_integrate_pc(normalized: bool, num_products: int):
    compiler = TorchCompiler()
    num_variables, num_channels = 5, 1
    scs = []
    last_sc = None
    for i in range(num_products):
        sci = build_simple_pc(num_variables, 4 + i, 3 + i, normalized=normalized)
        if i == 0:
            last_sc = sci
        else:
            last_sc = SF.multiply(last_sc, sci)
        scs.append(sci)
    tc: TensorizedCircuit = compiler.compile(last_sc)
    tcs: List[TensorizedCircuit] = [compiler.get_compiled_circuit(sc) for sc in scs]
    int_sc = SF.integrate(last_sc)
    int_tc = compiler.compile(int_sc)

    z = int_tc()  # compute the partition function
    assert z.shape == (1, 1)
    z = z.squeeze()
    if normalized:
        assert 0.0 < z.item() < 1.0
    else:
        assert not isclose(z.item(), 1.0)

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2 ** num_variables, 1, num_variables)
    scores = tc(worlds)
    assert scores.shape == (2 ** num_variables, 1, 1)
    scores = scores.squeeze()
    assert isclose(torch.sum(scores, dim=0), z)

    each_tc_scores = torch.prod(torch.stack([tci(worlds).squeeze() for tci in tcs], dim=0), dim=0)
    assert allclose(each_tc_scores, scores)
