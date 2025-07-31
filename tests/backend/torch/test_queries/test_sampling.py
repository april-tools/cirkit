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
    itertools.product([False, True], [False, True]),
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
    _, samples = query(num_samples=num_samples)
    assert samples.shape == (num_samples, tc.num_variables)

    # Map samples to indices of the probabilities computed above
    samples_idx = samples * torch.tensor(list(reversed([2**i for i in range(tc.num_variables)])))
    samples_idx = torch.sum(samples_idx, dim=-1)

    # Compute ratios and compare with the probabilities
    _, counts = torch.unique(samples_idx, return_counts=True)
    ratios = counts / num_samples
    assert allclose(ratios, probs, rtol=3e-2)


@pytest.mark.parametrize(
    "fold,optimize",
    itertools.product([False, True], [False, True]),
)
def test_query_unconditional_sampling_gaussian(fold: bool, optimize: bool):
    compiler = TorchCompiler(semiring="lse-sum", fold=fold, optimize=optimize)
    
    from cirkit.symbolic.circuit import Circuit, Scope
    from cirkit.symbolic.layers import GaussianLayer, HadamardLayer, SumLayer
    from cirkit.templates import utils

    # This parametrizes the mixture weights such that they add up to one.
    weight_factory = utils.parameterization_to_factory(utils.Parameterization(
        activation='softmax',   # Parameterize the sum weights by using a softmax activation
        initialization='uniform' # Initialize the sum weights by sampling from a standard normal distribution
    ))

    # We introduce one more mixture than in the original model
    # Again, SGD/Adam is not the best way to fit a (shallow) Gaussian mixture model
    units = 2+1 
    
    g0 = GaussianLayer(Scope((0,)), units)
    g1 = GaussianLayer(Scope((1,)), units)
    prod = HadamardLayer(num_input_units=units, arity=2)
    sl = SumLayer(units, 1, 1, weight_factory=weight_factory)

    sc = Circuit(
        layers=[g0, g1, prod, sl],  # Layers that appear in the circuit (i.e. nodes in the graph)
        in_layers={  # Connections between layers (i.e. edges in the graph as an adjacency list)
            g0: [],
            g1: [],
            prod: [g0, g1],
            sl: [prod],
        },
        outputs=[sl]  # Nodes that are returned by the circuit
    )
    
    tc: TorchCircuit = compiler.compile(sc)

    # Sample data points unconditionally
    num_samples = 1_000_000
    query = SamplingQuery(tc)
    # samples: (num_samples, D)
    _, samples = query(num_samples=num_samples)


@pytest.mark.parametrize(
    "fold,optimize",
    itertools.product([False, True], [False, True]),
)
def test_query_conditional_sampling(fold: bool, optimize: bool):
    compiler = TorchCompiler(semiring="lse-sum", fold=fold, optimize=optimize)
    sc = build_multivariate_monotonic_structured_cpt_pc(
        num_units=2, input_layer="bernoulli", parameterize=True, normalized=True
    )
    tc: TorchCircuit = compiler.compile(sc)

    evidence = torch.randint(0, 1, size=(1, tc.num_variables))
    evidence_vars = torch.zeros_like(evidence).bool()
    evidence_vars[:, 0] = True

    # Sample data points unconditionally
    num_samples = 1_000_000
    query = SamplingQuery(tc)
    # samples: (num_samples, D)
    _, samples = query(
        num_samples=num_samples,
        x=evidence,
        evidence_vars=evidence_vars
    )
    assert samples.shape == (num_samples, tc.num_variables)