import itertools

import pytest
import torch

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.queries import SamplingQuery
from tests.floats import allclose
from tests.symbolic.test_utils import (
    build_bivariate_monotonic_structured_cpt_pc,
    build_multivariate_monotonic_structured_cpt_pc,
)


@pytest.mark.parametrize(
    "fold,optimize",
    itertools.product([True], [False]),
)
def test_query_unconditional_sampling(fold: bool, optimize: bool):
    compiler = TorchCompiler(semiring="lse-sum", fold=fold, optimize=optimize)
    sc = build_multivariate_monotonic_structured_cpt_pc(
        num_units=2, input_layer="bernoulli", parameterize=True, normalized=True
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

    # Sample data points unconditionally
    num_samples = 1_000_000
    query = SamplingQuery(tc)
    # samples: (num_samples, D)
    samples, _ = query(num_samples=num_samples)
    assert samples.shape == (num_samples, tc.num_variables)

    # Map samples to indices of the probabilities computed above
    samples_idx = samples * torch.tensor(list(reversed([2**i for i in range(tc.num_variables)])))
    samples_idx = torch.sum(samples_idx, dim=-1)

    # Compute ratios and compare with the probabilities
    _, counts = torch.unique(samples_idx, return_counts=True)
    ratios = counts / num_samples
    assert allclose(ratios, probs, rtol=3e-2)
