import itertools
import tempfile

import numpy as np
import pytest
import torch
from scipy import integrate

import cirkit.symbolic.functional as SF
from cirkit.backend.torch.circuits import TorchCircuit, TorchConstantCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.parameters import Parameter, TensorParameter
from tests.floats import allclose, isclose
from tests.symbolic.test_utils import build_simple_pc


@pytest.mark.parametrize(
    "fold,optimize",
    itertools.product([False, True], [False, True]),
)
def test_save_load_statedict(fold: bool, optimize: bool):
    num_variables = 6
    compiler = TorchCompiler(fold=fold, optimize=optimize)
    sc = build_simple_pc(num_variables, 3, 4, num_repetitions=2)
    tc: TorchCircuit = compiler.compile(sc)

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
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
