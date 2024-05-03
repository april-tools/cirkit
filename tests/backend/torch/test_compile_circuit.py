import itertools

import numpy as np
import pytest
import torch
from scipy import integrate

import cirkit.symbolic.functional as SF
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.models import TorchCircuit, TorchConstantCircuit
from tests.floats import allclose, isclose
from tests.symbolic.test_utils import build_simple_circuit, build_simple_pc


@pytest.mark.parametrize(
    "fold,num_variables,num_input_units,num_sum_units,num_repetitions",
    itertools.product([False, True], [1, 8], [1, 4], [1, 3], [1]),
)
def test_compile_output_shape(
    fold: bool, num_variables: int, num_input_units: int, num_sum_units: int, num_repetitions: int
):
    compiler = TorchCompiler(fold=fold)
    sc = build_simple_circuit(
        num_variables, num_input_units, num_sum_units, num_repetitions=num_repetitions
    )
    tc: TorchCircuit = compiler.compile(sc)

    batch_size = 42
    input_shape = (batch_size, 1, num_variables)
    x = torch.zeros(input_shape)
    y = tc(x)
    assert y.shape == (batch_size, 1, 1)
    assert torch.all(torch.isfinite(y))


@pytest.mark.parametrize(
    "fold,semiring,num_variables,normalized",
    itertools.product([False, True], ["lse-sum", "sum-product"], [1, 2, 5], [False, True]),
)
def test_compile_integrate_pc_discrete(
    fold: bool, semiring: str, num_variables: int, normalized: bool
):
    compiler = TorchCompiler(fold=fold, semiring=semiring)
    sc = build_simple_pc(num_variables, 4, 3, num_repetitions=3, normalized=normalized)

    int_sc = SF.integrate(sc)
    int_tc: TorchConstantCircuit = compiler.compile(int_sc)
    assert isinstance(int_tc, TorchConstantCircuit)
    tc: TorchCircuit = compiler.get_compiled_circuit(sc)
    assert isinstance(tc, TorchCircuit)

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


@pytest.mark.slow
@pytest.mark.parametrize(
    "semiring,normalized",
    itertools.product(["lse-sum", "sum-product"], [False, True]),
)
def test_compile_integrate_pc_continuous(semiring: str, normalized: bool):
    compiler = TorchCompiler(semiring=semiring)
    num_variables = 2
    sc = build_simple_pc(num_variables, input_layer="gaussian", normalized=normalized)

    int_sc = SF.integrate(sc)
    int_tc: TorchConstantCircuit = compiler.compile(int_sc)
    assert isinstance(int_tc, TorchConstantCircuit)
    tc: TorchCircuit = compiler.get_compiled_circuit(sc)
    assert isinstance(tc, TorchCircuit)

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

    # Test the integral of the circuit (using a quadrature rule)
    if semiring == "sum-product":
        df = lambda y, x: tc(torch.Tensor([[[x, y]]])).squeeze()
    elif semiring == "lse-sum":
        df = lambda y, x: torch.exp(tc(torch.Tensor([[[x, y]]]))).squeeze()
    else:
        assert False
    int_a, int_b = -np.inf, np.inf
    ig, err = integrate.dblquad(df, int_a, int_b, int_a, int_b)
    if normalized:
        assert np.isclose(ig, 1.0, atol=1e-15)
    elif semiring == "sum-product":
        assert np.isclose(ig, z.item(), atol=1e-15)
    elif semiring == "lse-sum":
        assert np.isclose(ig, torch.exp(z).item(), atol=1e-15)
    else:
        assert False


@pytest.mark.parametrize(
    "fold,semiring,normalized,num_variables,num_products",
    itertools.product(
        [False, True], ["sum-product", "lse-sum"], [False, True], [1, 2, 5], [2, 3, 4]
    ),
)
def test_compile_product_integrate_pc_discrete(
    fold: bool, semiring: str, normalized: bool, num_variables: int, num_products: int
):
    compiler = TorchCompiler(fold=fold, semiring=semiring)
    scs, tcs = [], []
    last_sc = None
    for i in range(num_products):
        sci = build_simple_pc(num_variables, 4 + i, 3 + i, normalized=normalized)
        tci = compiler.compile(sci)
        scs.append(sci)
        tcs.append(tci)
        last_sc = sci if i == 0 else SF.multiply(last_sc, sci)
    tc: TorchCircuit = compiler.compile(last_sc)
    int_sc = SF.integrate(last_sc)
    int_tc = compiler.compile(int_sc)

    # Test the partition function value
    z = int_tc()
    assert z.shape == (1, 1)
    z = z.squeeze()
    if normalized:
        if semiring == "sum-product":
            assert 0.0 < z.item() < 1.0
        elif semiring == "lse-sum":
            assert -np.inf < z.item() < 0.0
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

    # Test the products of the circuits evaluated over all possible assignments
    each_tc_scores = torch.stack([tci(worlds).squeeze() for tci in tcs], dim=0)
    if semiring == "sum-product":
        assert allclose(torch.prod(each_tc_scores, dim=0), scores)
    elif semiring == "lse-sum":
        assert allclose(torch.sum(each_tc_scores, dim=0), scores)
    else:
        assert False


# @pytest.mark.slow
# @pytest.mark.parametrize(
#     "semiring,normalized,num_products",
#     itertools.product(["lse-sum", "sum-product"], [False, True], [2, 3]),
# )
# def test_compile_product_integrate_pc_continuous(
#     semiring: str, normalized: bool, num_products: int
# ):
#     num_variables = 2
#     compiler = TorchCompiler(semiring=semiring)
#     scs, tcs = [], []
#     last_sc = None
#     for i in range(num_products):
#         sci = build_simple_pc(
#             num_variables, 4 + i, 3 + i, input_layer="gaussian", normalized=normalized
#         )
#         tci = compiler.compile(sci)
#         scs.append(sci)
#         tcs.append(tci)
#         last_sc = sci if i == 0 else SF.multiply(last_sc, sci)
#     tc: TensorizedCircuit = compiler.compile(last_sc)
#
#     # Test the product of the circuits evaluated over some randomly-chosen points
#     worlds = torch.randn(64, 1, num_variables)
#     scores = tc(worlds).squeeze()
#     each_tc_scores = torch.stack([tci(worlds).squeeze() for tci in tcs], dim=0)
#     if semiring == "sum-product":
#         assert allclose(torch.prod(each_tc_scores, dim=0), scores)
#     elif semiring == "lse-sum":
#         assert allclose(torch.sum(each_tc_scores, dim=0), scores)
#     else:
#         assert False
