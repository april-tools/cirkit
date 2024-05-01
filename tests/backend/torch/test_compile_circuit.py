import itertools

import numpy as np
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


@pytest.mark.parametrize(
    "semiring,num_variables,normalized",
    itertools.product(["lse-sum", "sum-product"], [1, 2, 5], [False, True]),
)
def test_compile_integrate_pc(semiring: str, num_variables: int, normalized: bool):
    compiler = TorchCompiler(semiring=semiring)
    sc = build_simple_pc(num_variables, 4, 3, num_repetitions=3, normalized=normalized)

    int_sc = SF.integrate(sc)
    int_tc: TensorizedConstantCircuit = compiler.compile(int_sc)
    assert isinstance(int_tc, TensorizedConstantCircuit)
    tc: TensorizedCircuit = compiler.get_compiled_circuit(sc)
    assert isinstance(tc, TensorizedCircuit)

    # Test the partition function value
    z = int_tc()
    assert z.shape == (1, 1)
    z = z.squeeze()
    if normalized:
        if semiring == "sum-product":
            assert isclose(z.item(), 1.0)
        elif semiring == "lse-sum":
            assert isclose(z.item(), 0.0)
        else:
            assert False
    else:
        if semiring == "sum-product":
            assert not isclose(z.item(), 1.0)
        elif semiring == "lse-sum":
            assert not isclose(z.item(), 0.0)
        else:
            assert False

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)
    scores = tc(worlds)
    assert scores.shape == (2**num_variables, 1, 1)
    scores = scores.squeeze()
    if semiring == "sum-product":
        assert isclose(torch.sum(scores), z)
    elif semiring == "lse-sum":
        assert isclose(torch.logsumexp(scores, dim=0), z)
    else:
        assert False


@pytest.mark.parametrize(
    "semiring,normalized,num_variables,num_products",
    itertools.product(["sum-product", "lse-sum"], [False, True], [1, 2, 5], [2, 3]),
)
def test_compile_product_integrate_pc(
    semiring: str, normalized: bool, num_variables: int, num_products: int
):
    compiler = TorchCompiler(semiring=semiring)
    scs, tcs = [], []
    last_sc = None
    for i in range(num_products):
        sci = build_simple_pc(num_variables, 4 + i, 3 + i, normalized=normalized)
        tci = compiler.compile(sci)
        scs.append(sci)
        tcs.append(tci)
        last_sc = sci if i == 0 else SF.multiply(last_sc, sci)
    tc: TensorizedCircuit = compiler.compile(last_sc)
    int_sc = SF.integrate(last_sc)
    int_tc = compiler.compile(int_sc)

    z = int_tc()  # compute the partition function
    assert z.shape == (1, 1)
    z = z.squeeze()
    if normalized:
        if semiring == "sum-product":
            assert 0.0 < z.item() < 1.0
        elif semiring == "lse-sum":
            assert -np.inf < z.item() < 0.0
        else:
            assert False
    else:
        if semiring == "sum-product":
            assert not isclose(z.item(), 1.0)
        elif semiring == "lse-sum":
            assert not isclose(z.item(), 0.0)
        else:
            assert False

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)
    scores = tc(worlds)
    assert scores.shape == (2**num_variables, 1, 1)
    scores = scores.squeeze()
    if semiring == "sum-product":
        assert isclose(torch.sum(scores, dim=0), z)
    elif semiring == "lse-sum":
        assert isclose(torch.logsumexp(scores, dim=0), z)
    else:
        assert False

    each_tc_scores = torch.stack([tci(worlds).squeeze() for tci in tcs], dim=0)

    if semiring == "sum-product":
        assert allclose(torch.prod(each_tc_scores, dim=0), scores)
    elif semiring == "lse-sum":
        assert allclose(torch.sum(each_tc_scores, dim=0), scores)
    else:
        assert False
