import itertools
import tempfile

import pytest
import torch

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from tests.floats import allclose
from tests.symbolic.test_utils import build_monotonic_structured_categorical_cpt_pc


@pytest.mark.parametrize(
    "semiring,fold,optimize",
    itertools.product(["sum-product", "lse-sum"], [False, True], [False, True]),
)
def test_save_load_statedict(semiring: str, fold: bool, optimize: bool):
    compiler = TorchCompiler(semiring=semiring, fold=fold, optimize=optimize)
    sc = build_monotonic_structured_categorical_cpt_pc(return_ground_truth=False)
    tc: TorchCircuit = compiler.compile(sc)

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=tc.num_variables))).unsqueeze(
        dim=-2
    )
    scores = tc(worlds)
    assert scores.shape == (len(worlds), 1, 1)

    state_dict_filepath = tempfile.NamedTemporaryFile().name
    torch.save(tc.state_dict(), state_dict_filepath)
    del tc
    tc: TorchCircuit = compiler.compile(sc)
    tc.load_state_dict(torch.load(state_dict_filepath, weights_only=True))
    checkpoint_scores = tc(worlds)
    assert checkpoint_scores.shape == (len(worlds), 1, 1)
    assert allclose(scores, checkpoint_scores)
