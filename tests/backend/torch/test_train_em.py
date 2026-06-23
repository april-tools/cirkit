import itertools
import tempfile

import pytest
import torch

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from tests.floats import allclose
from tests.symbolic.test_utils import build_monotonic_structured_categorical_cpt_pc

import torch.distributions as D
import math
from cirkit.symbolic.circuit import Circuit, Scope
from cirkit.symbolic.layers import (
    CategoricalLayer,
    GaussianLayer,
    SumLayer,
    HadamardLayer,
)
from cirkit.templates import utils
from cirkit.backend.torch.em_optimizer import EM


def build_gaus_symbolic_circuit(units) -> Circuit:
    weight_factory = utils.parameterization_to_factory(
        utils.Parameterization(
            activation="none",  # Parameterize the sum weights by using a softmax activation
            initialization="uniform",  # Initialize the sum weights by sampling from a standard normal distribution
            initialization_kwargs={"convex": True},
        )
    )

    mean_factory = utils.parameterization_to_factory(
        utils.Parameterization(
            activation="none",  # Parameterize the sum weights by using a softmax activation
            initialization="uniform",  # Initialize the sum weights by sampling from a standard normal distribution
        )
    )

    g0 = GaussianLayer(
        Scope((0,)), units, mean_factory=mean_factory, stddev_factory=mean_factory
    )
    g1 = GaussianLayer(
        Scope((1,)), units, mean_factory=mean_factory, stddev_factory=mean_factory
    )
    prod = HadamardLayer(num_input_units=units, arity=2)
    sl = SumLayer(units, 1, 1, weight_factory=weight_factory)

    return Circuit(
        layers=[
            g0,
            g1,
            prod,
            sl,
        ],  # Layers that appear in the circuit (i.e. nodes in the graph)
        in_layers={  # Connections between layers (i.e. edges in the graph as an adjacency list)
            g0: [],
            g1: [],
            prod: [g0, g1],
            sl: [prod],
        },
        outputs=[sl],  # Nodes that are returned by the circuit
    )


def test_train_em_gaussian_pc():
    torch.set_grad_enabled(True)
    assert torch.is_grad_enabled()
    radius = 2  # Distance of the centers from the origin
    K = 8  # Number of clusters
    mus = (
        torch.tensor(
            [
                [math.cos(2 * math.pi * n / K) for n in range(K)],
                [math.sin(2 * math.pi * n / K) for n in range(K)],
            ]
        ).T
        * radius
    )
    sigma = 0.2  # Standard deviatiomix = D.Categorical(torch.ones(K,))
    comp = D.Independent(D.Normal(mus, sigma), 1)
    mix = D.Categorical(
        torch.ones(
            K,
        )
    )
    gmm = D.MixtureSameFamily(mix, comp)

    dataset = gmm.sample((1000,))
    sc = build_gaus_symbolic_circuit(K + 1)
    compiler = TorchCompiler(semiring="lse-sum", fold=True, optimize=True)
    cc = compiler.compile(sc)
    cc = cc.train()

    optim = EM(cc, lr=1, pseudocount=0.0)
    losses = []
    for epoch in range(100):
        ll = cc(dataset)
        loss = ll.mean()
        loss.backward()
        optim.step()
        cc.zero_grad()

        losses.append(-loss)

    assert sorted(losses, reverse=True) == losses, "Loss should be decreasing"


def build_cat_symbolic_circuit(units, n_cat) -> Circuit:
    # This parametrizes the mixture weights such that they add up to one.
    weight_factory = utils.parameterization_to_factory(
        utils.Parameterization(
            activation="none",  # Parameterize the sum weights by using a softmax activation
            initialization="uniform",  # Initialize the sum weights by sampling from a standard normal distribution
            initialization_kwargs={"convex": True},
        )
    )

    c0 = CategoricalLayer(
        Scope((0,)), units, num_categories=n_cat, probs_factory=weight_factory
    )
    c1 = CategoricalLayer(
        Scope((1,)), units, num_categories=n_cat, probs_factory=weight_factory
    )
    prod = HadamardLayer(num_input_units=units, arity=2)
    sl = SumLayer(units, 1, 1, weight_factory=weight_factory)

    return Circuit(
        layers=[
            c0,
            c1,
            prod,
            sl,
        ],  # Layers that appear in the circuit (i.e. nodes in the graph)
        in_layers={  # Connections between layers (i.e. edges in the graph as an adjacency list)
            c0: [],
            c1: [],
            prod: [c0, c1],
            sl: [prod],
        },
        outputs=[sl],  # Nodes that are returned by the circuit
    )


def test_train_em_categorical_pc():
    torch.set_grad_enabled(True)
    assert torch.is_grad_enabled()
    mix_probs = torch.tensor([0.3, 0.7])
    M = 2  # number of mixture components
    S = 2  # dimensionality of each sample
    K = 4  # categories per dimension
    component_logits = torch.randn(M, S, K)

    comp = D.Independent(D.Categorical(logits=component_logits), 1)
    mix = D.Categorical(probs=mix_probs)

    gmm = D.MixtureSameFamily(mix, comp)

    dataset = gmm.sample((1000,))
    sc = build_cat_symbolic_circuit(M + 1, K)
    compiler = TorchCompiler(semiring="lse-sum", fold=True, optimize=True)
    cc = compiler.compile(sc)
    cc = cc.train()

    optim = EM(cc, lr=1, pseudocount=0.0)
    losses = []
    for epoch in range(100):
        ll = cc(dataset)
        loss = ll.mean()
        loss.backward()
        optim.step()
        cc.zero_grad()

        losses.append(-loss)

    assert sorted(losses, reverse=True) == losses, "Loss should be decreasing"


def build_cat_symbolic_circuit_test_update(n_cat) -> Circuit:
    weight_factory = utils.parameterization_to_factory(
        utils.Parameterization(
            activation="none",  # Parameterize the sum weights with no activation
            initialization="uniform",  # Initialize the sum weights by sampling from a uniform distribution
            initialization_kwargs={"convex": True}, # This initializes the mixture weights such that they add up to one.
        )
    )

    c0 = CategoricalLayer(
        Scope((0,)), 1, num_categories=n_cat, probs_factory=weight_factory
    )
    c1 = CategoricalLayer(
        Scope((1,)), 1, num_categories=n_cat, probs_factory=weight_factory
    )
    c2 = CategoricalLayer(
        Scope((0,)), 1, num_categories=n_cat, probs_factory=weight_factory
    )
    c3 = CategoricalLayer(
        Scope((1,)), 1, num_categories=n_cat, probs_factory=weight_factory
    )
    prod0 = HadamardLayer(num_input_units=1, arity=2)
    prod1 = HadamardLayer(num_input_units=1, arity=2)
    sl = SumLayer(1, 1, 2, weight_factory=weight_factory)

    return Circuit(
        layers=[
            c0,
            c1,
            c2,
            c3,
            prod0,
            prod1,
            sl,
        ],  # Layers that appear in the circuit (i.e. nodes in the graph)
        in_layers={  # Connections between layers (i.e. edges in the graph as an adjacency list)
            c0: [],
            c1: [],
            prod0: [c0, c1],
            prod1: [c2, c3],
            sl: [prod0, prod1],
        },
        outputs=[sl],  # Nodes that are returned by the circuit
    )


def test_em_update_categorical_pc():
    torch.set_grad_enabled(True)
    assert torch.is_grad_enabled()

    N_CAT = 3 # Each variable X0, X1 has 3 categories (0, 1, 2)
    sc = build_cat_symbolic_circuit_test_update(N_CAT)
    compiler = TorchCompiler(semiring="lse-sum", fold=True, optimize=True)
    cc = compiler.compile(sc)
    cc = cc.train()

    # Custom parameters so that we know the parameters after the update
    cat0 = torch.tensor([0.3, 0.3, 0.4])
    cat1 = torch.tensor([0.1, 0.2, 0.7])
    cat2 = torch.tensor([0.4, 0.4, 0.2])
    cat3 = torch.tensor([0.6, 0.1, 0.3])

    cat_probs = torch.vstack((cat0, cat1, cat2, cat3))
    sum_weights = torch.tensor([0.3, 0.7])

    # Manually set our custom parameters
    shape_probs = list(filter(lambda x: hasattr(x, '_ptensor'), cc.layers[0].modules()))[0]._ptensor.data.shape
    list(filter(lambda x: hasattr(x, '_ptensor'), cc.layers[0].modules()))[0]._ptensor.data = cat_probs.clone().reshape(shape_probs)

    shape_weights = list(filter(lambda x: hasattr(x, '_ptensor'), cc.layers[2].modules()))[0]._ptensor.data.shape
    list(filter(lambda x: hasattr(x, '_ptensor'), cc.layers[2].modules()))[0]._ptensor.data = sum_weights.clone().reshape(shape_weights)

    # Single training instance (X0=0, X1=2)
    train_instance = torch.tensor([[0.0, 2.0]])

    # Full batch EM with lr=1 and no smoothing on our single training example
    optim = EM(cc, lr=1, pseudocount=0.0)

    ll = cc(train_instance)
    loss = ll.mean()
    loss.backward()
    optim.step()
    cc.zero_grad()

    # Compare updated parameters to their expected update
    # Updates are computed manually using the formulas in the [Einsum Networks](https://arxiv.org/abs/2004.06231v2) paper
    
    # With a single training instance, probabilities are replaced with the observed counts
    new_probs = list(filter(lambda x: hasattr(x, '_ptensor'), cc.layers[0].modules()))[0]._ptensor.data.clone()

    expected_probs = torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0]
         ]).reshape(shape_probs)
    
    assert allclose(new_probs, expected_probs), "Probabilities have not been updated as expected"
    
    # The circuit should compute p(x0=0, x1=2) = 0.3 * 0.3 * 0.7 + 0.7 * 0.4 * 0.3 = 0.147
    likelihood = loss.detach().exp() # Should be 0.147
    expected_likelihood = 0.147
    assert allclose(likelihood, expected_likelihood), "Likelihood has not been computed as expected"

    # Following the notation of the Einsum nets paper,
    # the sum weights should be updated for a single training instance as follows.
    # First we compute intermediate values n0, n1
    # n0 = p0(x0=0) * p1(x1=2) / c(x0=0, x1=2) = 0.3 * 0.7 / 0.147
    # n1 = p2(x0=0) * p3(x1=2) / c(x0=0, x1=2) = 0.4 * 0.3 / 0.147
    # Then the denominator for weight updates is:
    # D = w0 * n0 + w1 * n1 = 0.3 * 0.3 * 0.7 / 0.147 + 0.7 * 0.4 * 0.3 / 0.147 = 0.147 / 0.147 = 1.0
    # Overall, weights are then updated by:
    # w0 <- w0 * n0 / D = 0.3 * 0.3 * 0.7 / 0.147 = 0.4285...
    # w1 <- w1 * n1 / D = 0.7 * 0.4 * 0.3 / 0.147 = 0.5714...
    new_sum_weights = list(filter(lambda x: hasattr(x, '_ptensor'), cc.layers[2].modules()))[0]._ptensor.data.clone()
    expected_weight_0 = 0.063 / 0.147 # Should be approx 0.4286
    expected_weight_1 = 0.084 / 0.147 # Should be approx 0.5714
    expected_sum_weights = torch.tensor([expected_weight_0, expected_weight_1]).reshape(shape_weights)

    assert allclose(new_sum_weights, expected_sum_weights), "Sum weights have not been updated as expected"
