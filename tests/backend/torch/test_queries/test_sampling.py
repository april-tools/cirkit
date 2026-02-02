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
from cirkit.templates import data_modalities, utils
from cirkit.backend.torch.semiring import SumProductSemiring, LSESumSemiring

@pytest.mark.parametrize(
    "fold,optimize,semiring",
    itertools.product([False, True], [False, True], ["sum-product", "lse-sum"]),
)
def test_query_unconditional_sampling(fold: bool, optimize: bool, semiring: str):
    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    sc = build_multivariate_monotonic_structured_cpt_pc(
        num_units=2, input_layer="bernoulli", parameterize=True, normalized=True
    )
    tc: TorchCircuit = compiler.compile(sc)

    # Compute the probabilities
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=tc.num_variables)))
    assert worlds.shape == (2**tc.num_variables, tc.num_variables)
    tc_outputs = LSESumSemiring.map_from(tc(worlds), compiler.semiring)
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
    samples = samples.squeeze(dim=1)

    # Map samples to indices of the probabilities computed above
    samples_idx = samples * torch.tensor(list(reversed([2**i for i in range(tc.num_variables)])))
    samples_idx = torch.sum(samples_idx, dim=-1)

    # Compute ratios and compare with the probabilities
    _, counts = torch.unique(samples_idx, return_counts=True)
    ratios = counts / num_samples
    assert allclose(ratios, probs, atol=3e-2)


@pytest.mark.parametrize(
    "fold,optimize,semiring",
    itertools.product([False, True], [False, True], ["sum-product", "lse-sum"]),
)
def test_query_unconditional_sampling_image_pc(fold: bool, optimize: bool, semiring: str):
    sc = data_modalities.image_data(
        (1, 3, 3),                # The shape of MNIST image, i.e., (num_channels, image_height, image_width)
        region_graph='quad-graph',  # Select the structure of the circuit to follow the QuadGraph region graph
        input_layer='gaussian',     # Use Categorical distributions for the pixel values (0-255) as input layers
        num_input_units=15,         # Each input layer consists of 15 Categorical input units
        sum_product_layer='cp',     # Use CP sum-product layers, i.e., alternate dense layers with Hadamard product layers
        num_sum_units=7,           # Each dense sum layer 7 of 64 sum units
        sum_weight_param=utils.Parameterization(
            activation='softmax',   # Parameterize the sum weights by using a softmax activation
            initialization='normal' # Initialize the sum weights by sampling from a standard normal distribution
        )
    )

    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    tc = compiler.compile(sc)

    sampling_query = SamplingQuery(tc)
    samples, _ = sampling_query(num_samples=10)
    
    # This test is only checking the tensor shape for now.
    assert samples.shape == torch.Size([10, 1*3*3])


@pytest.mark.parametrize(
    "fold,optimize,semiring",
    itertools.product([False, True], [False, True], ["sum-product", "lse-sum"]),
)
def test_query_conditional_sampling(fold: bool, optimize: bool, semiring: str):
    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    sc = build_multivariate_monotonic_structured_cpt_pc(
        num_units=2, input_layer="bernoulli", parameterize=True, normalized=True
    )
    tc: TorchCircuit = compiler.compile(sc)

    # Compute the probabilities
    worlds = torch.tensor(list(itertools.product([0, 1], repeat=tc.num_variables)))
    assert worlds.shape == (2**tc.num_variables, tc.num_variables)
    tc_outputs = LSESumSemiring.map_from(tc(worlds), compiler.semiring)
    assert tc_outputs.shape == (2**tc.num_variables, 1, 1)
    assert torch.all(torch.isfinite(tc_outputs))
    probs = torch.exp(tc_outputs)
    probs = probs.squeeze(dim=2).squeeze(dim=1)

    # Sample data points unconditionally
    num_samples = 1_000_00
    query = SamplingQuery(tc)
    # samples: (num_samples, D)
    samples, _ = query(num_samples=num_samples)
    assert samples.shape == (num_samples, tc.num_variables)
    samples = samples.squeeze(dim=1)


    evidence = samples.clone().to(dtype=torch.get_default_dtype())
    evidence[torch.randn_like(samples.float()) < 0.5] = float('nan')
    cond_samples, _ = query(num_samples=7, evidence=evidence)

    assert cond_samples.shape[0] == 7 and cond_samples.shape[1:] == evidence.shape
    assert torch.allclose(
        cond_samples[~evidence.isnan().unsqueeze(0).repeat(7,1,1)], 
        evidence[~evidence.isnan()].unsqueeze(0).repeat(7,1,1).flatten()
    )
    
    # Map samples to indices of the probabilities computed above
    mask = samples[:, 0] == 0
    samples_idx = samples * torch.tensor(list(reversed([2**i for i in range(tc.num_variables)])))
    samples_idx = torch.sum(samples_idx, dim=-1)
    samples_idx = samples_idx[mask]

    # Compute ratios and compare with the probabilities
    _, counts = torch.unique(samples_idx, return_counts=True)
    ratios = counts / mask.sum()

    evidence = [float('nan')] * 5
    evidence[0] = 0
    evidence = torch.tensor(evidence)[None,]
    samples, _ = query(num_samples=num_samples, evidence=evidence)
    samples = samples.flatten(end_dim=1)
    samples_idx = samples * torch.tensor(list(reversed([2**i for i in range(tc.num_variables)])))
    samples_idx = torch.sum(samples_idx, dim=-1)
    _, counts = torch.unique(samples_idx, return_counts=True)
    other_ratios = counts / samples.shape[0]
    
    assert allclose(ratios, other_ratios, atol=3e-2)


@pytest.mark.parametrize(
    "fold,optimize,semiring",
    itertools.product([False, True], [False, True], ["sum-product", "lse-sum"]),
)
def test_query_conditional_sampling_image_pc(fold: bool, optimize: bool, semiring: str):
    sc = data_modalities.image_data(
        (1, 3, 3),                # The shape of MNIST image, i.e., (num_channels, image_height, image_width)
        region_graph='quad-graph',  # Select the structure of the circuit to follow the QuadGraph region graph
        input_layer='gaussian',     # Use Categorical distributions for the pixel values (0-255) as input layers
        num_input_units=15,         # Each input layer consists of 15 Categorical input units
        sum_product_layer='cp',     # Use CP sum-product layers, i.e., alternate dense layers with Hadamard product layers
        num_sum_units=7,           # Each dense sum layer 7 of 64 sum units
        sum_weight_param=utils.Parameterization(
            activation='softmax',   # Parameterize the sum weights by using a softmax activation
            initialization='normal' # Initialize the sum weights by sampling from a standard normal distribution
        )
    )

    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    tc = compiler.compile(sc)

    sampling_query = SamplingQuery(tc)
    evidence = torch.randn((10, 3*3))
    evidence[evidence < 0] = float('nan')
    samples, _ = sampling_query(num_samples=3, evidence=evidence)

    # This test is only checking the tensor shape for now.
    assert samples.shape == torch.Size([3, 10, 1*3*3])
    assert torch.allclose(
        samples[~evidence.isnan().unsqueeze(0).repeat(3,1,1)], 
        evidence[~evidence.isnan()].unsqueeze(0).repeat(3,1,1).flatten()
    )
    
