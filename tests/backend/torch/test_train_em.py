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
    # This parametrizes the mixture weights such that they add up to one.
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

    optim = EM(cc, lr=1, pseudocount=1e-8)
    losses = []
    for epoch in range(100):
        ll = cc(dataset)
        loss = ll.mean()
        loss.backward()
        optim.step()
        cc.zero_grad()

        losses.append(-loss)

    assert sorted(losses, reverse=True) == losses, "Loss should be strictly decreasing"


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

    optim = EM(cc, lr=1, pseudocount=1e-8)
    losses = []
    for epoch in range(100):
        ll = cc(dataset)
        loss = ll.mean()
        loss.backward()
        optim.step()
        cc.zero_grad()

        losses.append(-loss)

    assert sorted(losses, reverse=True) == losses, "Loss should be strictly decreasing"
