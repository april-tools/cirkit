import itertools

import pytest
import torch

pytest.importorskip("extended_einsum")

import cirkit.symbolic.functional as SF
from cirkit.pipeline import PipelineContext
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, PolynomialLayer, SumLayer
from cirkit.utils.scope import Scope
from tests.symbolic.test_utils import build_multivariate_monotonic_structured_cpt_pc


def build_trained_pipeline() -> tuple:
    """Compile a symbolic circuit with the extended-einsum backend and train it for
    a few steps, so that the tests can check that circuits obtained by symbolic
    operators re-use the trained parameters."""
    torch.manual_seed(42)
    sc = build_multivariate_monotonic_structured_cpt_pc(num_units=3)
    ctx = PipelineContext(backend="xe-torch", semiring="lse-sum", fold=True, optimize=True)
    cc = ctx.compile(sc)
    optimizer = torch.optim.Adam(cc.parameters(), lr=0.1)
    data = torch.randint(2, size=(64, 5))
    with torch.enable_grad():
        for _ in range(8):
            optimizer.zero_grad()
            loss = -torch.mean(cc(data))
            loss.backward()
            optimizer.step()
    return sc, ctx, cc


def all_worlds(num_variables: int) -> torch.Tensor:
    return torch.tensor(list(itertools.product([0, 1], repeat=num_variables)))


def test_integrate_trained_circuit():
    _, ctx, cc = build_trained_pipeline()
    worlds = all_worlds(5)
    log_scores = cc(worlds).flatten()  # (32,)

    # The full integral must match the brute-force sum over all the worlds of the
    # trained circuit. Note that this fails if the integral circuit does not re-use
    # the trained parameters.
    int_cc = ctx.integrate(cc)
    assert int_cc is not cc
    log_partition = int_cc()
    assert log_partition.shape == (1, 1)
    assert torch.allclose(
        log_partition.flatten(), torch.logsumexp(log_scores, dim=0), rtol=1e-5, atol=1e-6
    )

    # Partially marginalizing one variable must match the brute-force sum over its
    # two possible values
    mar_cc = ctx.integrate(cc, scope=Scope([4]))
    x = worlds[worlds[:, 4] == 0]
    mar_scores = mar_cc(x).flatten()
    brute_force_scores = torch.logsumexp(
        torch.stack(
            [
                cc(torch.cat([x[:, :4], torch.full((x.shape[0], 1), v)], dim=1)).flatten()
                for v in (0, 1)
            ]
        ),
        dim=0,
    )
    assert torch.allclose(mar_scores, brute_force_scores, rtol=1e-5, atol=1e-6)


def test_evidence_trained_circuit():
    sc, ctx, cc = build_trained_pipeline()
    worlds = all_worlds(5)
    log_scores = cc(worlds).flatten()

    obs = {0: 1, 1: 0, 2: 1, 3: 0, 4: 1}
    ev_cc = ctx.compile(SF.evidence(sc, obs))
    assert not ev_cc.scope
    ev_score = ev_cc()
    assert ev_score.shape == (1, 1)
    world_index = int("".join(str(obs[i]) for i in range(5)), base=2)
    assert torch.allclose(ev_score.flatten(), log_scores[world_index], rtol=1e-5, atol=1e-6)


def test_multiply_trained_circuit():
    _, ctx, cc = build_trained_pipeline()
    worlds = all_worlds(5)
    log_scores = cc(worlds).flatten()

    # The product of the circuit with itself computes its square
    prod_cc = ctx.multiply(cc, cc)
    prod_scores = prod_cc(worlds).flatten()
    assert torch.allclose(prod_scores, 2.0 * log_scores, rtol=1e-5, atol=1e-5)

    # ... and integrating the product circuit must match the brute-force sum of the
    # squared scores over all the worlds
    int_prod_cc = ctx.integrate(prod_cc)
    assert torch.allclose(
        int_prod_cc().flatten(),
        torch.logsumexp(2.0 * log_scores, dim=0),
        rtol=1e-5,
        atol=1e-5,
    )


def test_concatenate_trained_circuits():
    _, ctx, cc = build_trained_pipeline()
    worlds = all_worlds(5)
    log_scores = cc(worlds)  # (32, 1, 1)

    cat_cc = ctx.concatenate(cc, cc)
    cat_scores = cat_cc(worlds)
    assert cat_scores.shape == (32, 2, 1)
    assert torch.allclose(cat_scores, log_scores.expand(-1, 2, -1), rtol=1e-5, atol=1e-6)


def test_differentiate_polynomial_circuit():
    torch.manual_seed(42)
    layers = [
        PolynomialLayer(Scope([0]), 2, degree=2),
        PolynomialLayer(Scope([1]), 2, degree=2),
        HadamardLayer(2, arity=2),
        SumLayer(2, 1),
    ]
    sc = Circuit(
        layers,
        {layers[2]: [layers[0], layers[1]], layers[3]: [layers[2]]},
        outputs=[layers[3]],
    )
    ctx = PipelineContext(backend="xe-torch", semiring="sum-product", fold=True, optimize=True)
    cc = ctx.compile(sc)
    diff_cc = ctx.differentiate(cc)
    x = torch.randn(6, 2, dtype=torch.float64).to(torch.get_default_dtype())
    y = diff_cc(x)
    # The output contains the partial derivatives w.r.t. the two variables,
    # followed by the circuit output itself
    assert y.shape == (6, 3, 1)
    assert torch.allclose(y[:, 2], cc(x)[:, 0], rtol=1e-6, atol=1e-7)
    # Check the partial derivatives against finite differences
    eps = 2**-12
    for var in range(2):
        offset = torch.zeros(1, 2)
        offset[0, var] = eps
        finite_diff = (cc(x + offset)[:, 0] - cc(x - offset)[:, 0]) / (2.0 * eps)
        assert torch.allclose(y[:, var], finite_diff, rtol=1e-2, atol=1e-3)
