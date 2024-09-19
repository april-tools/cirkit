import itertools

import pytest
import torch

import cirkit.symbolic.functional as SF
from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.backend.torch.queries import IntegrateQuery
from cirkit.utils.scope import Scope
from tests.floats import allclose
from tests.symbolic.test_utils import build_monotonic_structured_categorical_cpt_pc


@pytest.mark.parametrize(
    "semiring,fold,optimize",
    itertools.product(["lse-sum", "sum-product"], [False, True], [False, True]),
)
def test_query_marginalize_monotonic_pc_categorical(semiring: str, fold: bool, optimize: bool):
    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    sc, gt_outputs, gt_partition_func = build_monotonic_structured_categorical_cpt_pc(
        return_ground_truth=True
    )

    mar_sc = SF.integrate(sc, scope=Scope([4]))
    mar_tc: TorchCircuit = compiler.compile(mar_sc)
    assert isinstance(mar_tc, TorchCircuit)
    tc: TorchCircuit = compiler.get_compiled_circuit(sc)
    assert isinstance(tc, TorchCircuit)

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
    mar_scores1 = mar_tc(mar_worlds)
    mar_query = IntegrateQuery(tc)
    mar_scores2 = mar_query(mar_worlds, integrate_vars=Scope([4]))
    assert mar_scores1.shape == mar_scores2.shape
    assert allclose(mar_scores1, mar_scores2)
