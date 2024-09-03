import itertools

import pytest
import torch

import cirkit.symbolic.functional as SF
from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.semiring import SumProductSemiring
from cirkit.utils.scope import Scope
from tests.floats import allclose, isclose
from tests.symbolic.test_utils import build_monotonic_structured_categorical_cpt_pc


@pytest.mark.parametrize(
    "semiring,fold,optimize",
    itertools.product(["lse-sum", "sum-product"], [False, True], [False, True]),
)
def test_marginalize_monotonic_pc_discrete(semiring: str, fold: bool, optimize: bool):
    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    sc, gt_outputs, gt_partition_func = build_monotonic_structured_categorical_cpt_pc(
        return_ground_truth=True
    )

    mar_sc = SF.integrate(sc, scope=Scope([4]))
    mar_tc: TorchCircuit = compiler.compile(mar_sc)
    assert isinstance(mar_tc, TorchCircuit)
    tc: TorchCircuit = compiler.get_compiled_circuit(sc)
    assert isinstance(tc, TorchCircuit)

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=tc.num_variables))).unsqueeze(
        dim=-2
    )
    scores = tc(worlds)
    assert scores.shape == (2**tc.num_variables, 1, 1)
    scores = scores.squeeze()

    mar_worlds = torch.cat(
        [
            torch.tensor(list(itertools.product([0, 1], repeat=tc.num_variables - 1))).unsqueeze(
                dim=-2
            ),
            torch.zeros(2 ** (tc.num_variables - 1), dtype=torch.int64)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1),
        ],
        dim=2,
    )
    mar_scores = mar_tc(mar_worlds)
    assert mar_scores.shape == (2 ** (tc.num_variables - 1), 1, 1)
    mar_scores = mar_scores.squeeze()
    assert allclose(compiler.semiring.sum(scores.view(-1, 2), dim=1), mar_scores)

    for x, y in gt_outputs["mar"].items():
        idx = int("".join(map(str, filter(lambda z: z != -1, x))), base=2)
        assert isclose(
            mar_scores[idx], compiler.semiring.map_from(torch.tensor(y), SumProductSemiring)
        ), f"Input: {x}"
