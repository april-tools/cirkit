import itertools

import pytest
import torch

import cirkit.symbolic.functional as SF
from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.utils.scope import Scope
from tests.floats import allclose, isclose
from tests.symbolic.test_utils import build_simple_pc


@pytest.mark.parametrize(
    "fold,semiring,num_variables,normalized",
    itertools.product([False, True], ["lse-sum", "sum-product"], [5], [False, True]),
)
def test_marginalize_pc_discrete(fold: bool, semiring: str, num_variables: int, normalized: bool):
    compiler = TorchCompiler(fold=fold, semiring=semiring)
    sc = build_simple_pc(num_variables, 3, 2, num_repetitions=3, normalized=normalized)

    mar_sc = SF.integrate(sc, scope=Scope([num_variables - 1]))
    mar_tc: TorchCircuit = compiler.compile(mar_sc)
    assert isinstance(mar_tc, TorchCircuit)
    tc: TorchCircuit = compiler.get_compiled_circuit(sc)
    assert isinstance(tc, TorchCircuit)

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)
    scores = tc(worlds)
    assert scores.shape == (2**num_variables, 1, 1)
    scores = scores.squeeze()

    mar_worlds = torch.cat(
        [
            torch.tensor(list(itertools.product([0, 1], repeat=num_variables - 1))).unsqueeze(
                dim=-2
            ),
            torch.zeros(2 ** (num_variables - 1), dtype=torch.int64)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1),
        ],
        dim=2,
    )
    mar_scores = mar_tc(mar_worlds)
    assert mar_scores.shape == (2 ** (num_variables - 1), 1, 1)
    mar_scores = mar_scores.squeeze()

    if semiring == "sum-product":
        assert allclose(torch.sum(scores.view(-1, 2), dim=1), mar_scores)
    elif semiring == "lse-sum":
        assert allclose(torch.logsumexp(scores.view(-1, 2), dim=1), mar_scores)
    else:
        assert False
