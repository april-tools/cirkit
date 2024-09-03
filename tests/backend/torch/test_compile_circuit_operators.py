import itertools

import numpy as np
import pytest
import torch

import cirkit.symbolic.functional as SF
from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from tests.floats import allclose, isclose
from tests.symbolic.test_utils import build_multivariate_monotonic_structured_cpt_pc


@pytest.mark.parametrize(
    "semiring,fold,optimize,normalized,num_products",
    itertools.product(
        ["sum-product", "lse-sum"], [False, True], [False, True], [False, True], [2, 3, 4]
    ),
)
def test_compile_product_integrate_pc_discrete(
    semiring: str, fold: bool, optimize: bool, normalized: bool, num_products: int
):
    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    scs, tcs = [], []
    last_sc = None
    for i in range(num_products):
        sci = build_multivariate_monotonic_structured_cpt_pc(num_units=2 + i, normalized=normalized)
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
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=tc.num_variables))).unsqueeze(
        dim=-2
    )
    scores = tc(worlds)
    assert scores.shape == (2**tc.num_variables, 1, 1)
    scores = scores.squeeze()
    assert isclose(compiler.semiring.sum(scores, dim=0), int_tc())

    # Test the products of the circuits evaluated over all possible assignments
    each_tc_scores = torch.stack([tci(worlds).squeeze() for tci in tcs], dim=0)
    assert allclose(compiler.semiring.prod(each_tc_scores, dim=0), scores)


# @pytest.mark.slow
# def test_compile_integrate_pc_continuous(fold: bool):
#     compiler = TorchCompiler(semiring="lse-sum", fold=True, optimize=True)
#     num_variables = 2
#     sc = build_simple_pc(num_variables, input_layer="gaussian")
#
#     int_sc = SF.integrate(sc)
#     int_tc: TorchConstantCircuit = compiler.compile(int_sc)
#     assert isinstance(int_tc, TorchConstantCircuit)
#     tc: TorchCircuit = compiler.get_compiled_circuit(sc)
#     assert isinstance(tc, TorchCircuit)
#
#     # Test the partition function value
#     z = int_tc()
#     assert z.shape == (1, 1)
#     z = z.squeeze()
#
#     # Test the integral of the circuit (using a quadrature rule)
#     df = lambda y, x: torch.exp(tc(torch.Tensor([[[x, y]]]))).squeeze()
#     int_a, int_b = -np.inf, np.inf
#     ig, err = integrate.dblquad(df, int_a, int_b, int_a, int_b)
#     assert np.isclose(ig, torch.exp(z).item(), atol=1e-15)
#
#
# @pytest.mark.slow
# @pytest.mark.parametrize(
#     "num_products",
#     [2, 3],
# )
# def test_compile_product_integrate_pc_continuous(num_products: int):
#     compiler = TorchCompiler(semiring="lse-sum", fold=True)
#     scs, tcs = [], []
#     last_sc = None
#     for i in range(num_products):
#         sci = build_simple_pc(2, 2 + i, 2, input_layer="gaussian")
#         tci = compiler.compile(sci)
#         scs.append(sci)
#         tcs.append(tci)
#         last_sc = sci if i == 0 else SF.multiply(last_sc, sci)
#     tc: TorchCircuit = compiler.compile(last_sc)
#     int_sc = SF.integrate(last_sc)
#     int_tc = compiler.compile(int_sc)
#
#     # Test the partition function value
#     z = int_tc()
#     assert z.shape == (1, 1)
#     z = z.squeeze()
#
#     # Test the integral of the circuit (using a quadrature rule)
#     df = lambda y, x: torch.exp(tc(torch.Tensor([[[x, y]]]))).squeeze()
#     int_a, int_b = -np.inf, np.inf
#     ig, err = integrate.dblquad(df, int_a, int_b, int_a, int_b)
#     assert np.isclose(ig, torch.exp(z).item(), atol=1e-15)
#
#     # Test the products of the circuits evaluated over all possible assignments
#     xs = torch.linspace(-5, 5, steps=16)
#     ys = torch.linspace(-5, 5, steps=16)
#     points = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=1).view(-1, 1, 2)
#     scores = tc(points)
#     scores = scores.squeeze()
#     each_tc_scores = torch.stack([tci(points).squeeze() for tci in tcs], dim=0)
#     assert allclose(torch.sum(each_tc_scores, dim=0), scores)
