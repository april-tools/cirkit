"""Check multichannel likelihoods, marginalization and shared parameter compilation."""

import itertools

import numpy as np
import pytest
import torch

from cirkit.pipeline import PipelineContext
from cirkit.symbolic import functional
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import MultichannelCategoricalLayer
from cirkit.symbolic.parameters import ConstantParameter, Parameter
from cirkit.utils.scope import Scope


@pytest.mark.parametrize(
    "semiring,fold,optimize,logits",
    tuple(
        itertools.product(
            ["sum-product", "lse-sum"],
            [False, True],
            [False, True],
            [False, True],
        )
    ),
)
def test_multichannel_likelihood_and_marginals(semiring, fold, optimize, logits):
    # Non-contiguous IDs also check that channel order is explicitly sorted.
    probs = np.array([[[0.2, 0.8], [0.3, 0.7], [0.6, 0.4]]])
    potentials = probs * 2 if logits else probs
    values = np.log(potentials) if logits else potentials
    param = Parameter.from_input(ConstantParameter(*values.shape, value=values))
    layer = MultichannelCategoricalLayer(
        Scope([9, 0, 5]),
        1,
        num_channels=3,
        num_categories=2,
        logits=param if logits else None,
        probs=param if not logits else None,
    )
    circuit = Circuit([layer], {}, [layer])
    ctx = PipelineContext(backend="torch", semiring=semiring, fold=fold, optimize=optimize)
    compiled = ctx.compile(circuit)
    states = torch.tensor(list(itertools.product([0, 1], repeat=3)))
    data = torch.zeros(8, 10, dtype=torch.long)
    data[:, [0, 5, 9]] = states
    expected = torch.tensor(potentials)[0, torch.arange(3), states].prod(dim=1)
    if semiring == "lse-sum":
        expected = expected.log()
    torch.testing.assert_close(compiled(data).flatten(), expected)

    marginal = ctx.compile(functional.integrate(circuit, scope=Scope([5])))
    remaining = torch.tensor(potentials)[0, [0, 2], states[:, [0, 2]]].prod(dim=1)
    remaining *= torch.tensor(potentials)[0, 1].sum()
    if semiring == "lse-sum":
        remaining = remaining.log()
    torch.testing.assert_close(marginal(data).flatten(), remaining)

    integrated = ctx.compile(functional.integrate(circuit))
    total = torch.tensor(potentials).sum(dim=2).prod().reshape(1)
    if semiring == "lse-sum":
        total = total.log()
    torch.testing.assert_close(integrated().flatten(), total)

    conjugated = ctx.compile(functional.conjugate(circuit))
    torch.testing.assert_close(conjugated(data), compiled(data))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_channels": 1},
        {"num_categories": 1},
        {"probs": Parameter.from_input(ConstantParameter(1, 2, 3))},
    ],
)
def test_multichannel_invalid_configuration(kwargs):
    config = {
        "scope": Scope([0, 1]),
        "num_output_units": 1,
        "num_channels": 2,
        "num_categories": 2,
    }
    config.update(kwargs)
    with pytest.raises(ValueError):
        MultichannelCategoricalLayer(**config)
