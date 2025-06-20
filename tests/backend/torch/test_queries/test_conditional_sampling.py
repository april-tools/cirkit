import itertools

import pytest
import torch

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.queries import ConditionalSamplingQuery
from cirkit.utils.scope import Scope
from tests.floats import allclose
from tests.symbolic.test_utils import (
    build_multivariate_monotonic_structured_cpt_pc,
)

@pytest.mark.parametrize(
    "fold,optimize",
    itertools.product([True, False], [True]),
)
def test_query_conditional_sampling(fold: bool, optimize: bool):
    compiler = TorchCompiler(semiring="lse-sum", fold=fold, optimize=optimize)
    sc = build_multivariate_monotonic_structured_cpt_pc(
        num_units=2, input_layer="bernoulli", parameterize=False, normalized=True
    )
    tc: TorchCircuit = compiler.compile(sc)

    # Compute the probabilities
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=tc.num_variables)))
    assert worlds.shape == (2**tc.num_variables, tc.num_variables)
    tc_outputs = tc(worlds)
    assert tc_outputs.shape == (2**tc.num_variables, 1, 1)
    assert torch.all(torch.isfinite(tc_outputs))
    probs = torch.exp(tc_outputs)
    probs = probs.squeeze(dim=2).squeeze(dim=1)
    # we want to calculate the conditional probabilities when 0th variable is set to one,
    # therefore will be marginalizing all other variables except the 0th
    vars_to_marginalize = [1, 2, 3, 4]
    # we are interested in the conditional probability when 0th var is set to 1.
    # therefore, extract the last 4 columns and re-normalize the probabilities.
    probs = probs[-2**len(vars_to_marginalize):] / torch.sum(probs[-2**len(vars_to_marginalize):])
    assert torch.allclose(torch.sum(probs), torch.tensor(1.0))

    # Sample data points conditionally
    num_samples = 1_000_000
    # convert marginalized variables to scopes
    scopes_to_marginalize = Scope(vars_to_marginalize)
    query = ConditionalSamplingQuery(tc)
    samples, _ = query(num_samples=num_samples, x=torch.as_tensor([1, 1, 0, 1, 0]).unsqueeze(0),
                       integrate_vars=scopes_to_marginalize)
    # extract the variables of the samples except 0th
    samples = samples[:, -len(scopes_to_marginalize):]

    # Map samples to indices of the probabilities computed above for variables except 0th
    samples_idx = samples * torch.tensor(list(reversed([2**i for i in range(tc.num_variables -
                                                                            (tc.num_variables -
                                                                             len(scopes_to_marginalize)))])))
    samples_idx = torch.sum(samples_idx, dim=-1)

    # Compute ratios and compare with the probabilities
    _, counts = torch.unique(samples_idx, return_counts=True)
    ratios = counts / num_samples
    assert allclose(ratios, probs, rtol=3e-2)
