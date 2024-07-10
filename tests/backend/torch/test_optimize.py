import itertools

import pytest
import torch

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.compiler import TorchCompiler
from tests.floats import allclose
from tests.symbolic.test_utils import build_simple_pc


@pytest.mark.parametrize("fold", [False, True])
def test_optimize_tucker(fold: bool):
    num_variables = 6
    sc = build_simple_pc(num_variables, 3, 2, sum_product_layer="tucker")

    unoptimized_compiler = TorchCompiler(fold=fold, optimize=False)
    unoptimized_tc: TorchCircuit = unoptimized_compiler.compile(sc)

    optimized_compiler = TorchCompiler(fold=fold, optimize=True)
    optimized_tc: TorchCircuit = optimized_compiler.compile(sc)

    if fold:
        pnames = [
            ("_nodes.0.logits._nodes.0._ptensor", "_nodes.0.logits._nodes.0._ptensor"),
            ("_nodes.1.logits._nodes.0._ptensor", "_nodes.1.logits._nodes.0._ptensor"),
            ("_nodes.2.weight._nodes.0._ptensor", "_nodes.2.weight._nodes.0._ptensor"),
            ("_nodes.4.weight._nodes.0._ptensor", "_nodes.3.weight._nodes.0._ptensor"),
            ("_nodes.6.weight._nodes.0._ptensor", "_nodes.4.weight._nodes.0._ptensor"),
        ]
    else:
        pnames = [
            ("_nodes.0.logits._nodes.0._ptensor", "_nodes.0.logits._nodes.0._ptensor"),
            ("_nodes.2.logits._nodes.0._ptensor", "_nodes.2.logits._nodes.0._ptensor"),
            ("_nodes.4.logits._nodes.0._ptensor", "_nodes.4.logits._nodes.0._ptensor"),
            ("_nodes.6.logits._nodes.0._ptensor", "_nodes.6.logits._nodes.0._ptensor"),
            ("_nodes.1.weight._nodes.0._ptensor", "_nodes.1.weight._nodes.0._ptensor"),
            ("_nodes.3.weight._nodes.0._ptensor", "_nodes.3.weight._nodes.0._ptensor"),
            ("_nodes.5.weight._nodes.0._ptensor", "_nodes.5.weight._nodes.0._ptensor"),
            ("_nodes.7.weight._nodes.0._ptensor", "_nodes.7.weight._nodes.0._ptensor"),
            ("_nodes.9.weight._nodes.0._ptensor", "_nodes.8.weight._nodes.0._ptensor"),
            ("_nodes.11.weight._nodes.0._ptensor", "_nodes.9.weight._nodes.0._ptensor"),
            ("_nodes.13.weight._nodes.0._ptensor", "_nodes.10.weight._nodes.0._ptensor"),
        ]

    for unoptimized_pname, optimized_pname in pnames:
        optimized_tc.load_state_dict(
            {optimized_pname: unoptimized_tc.state_dict()[unoptimized_pname]}, strict=False
        )

    worlds = torch.tensor(list(itertools.product([0, 1], repeat=num_variables))).unsqueeze(dim=-2)
    assert worlds.shape == (2**num_variables, 1, num_variables)

    unoptimized_scores = unoptimized_tc(worlds)
    assert unoptimized_scores.shape == (2**num_variables, 1, 1)

    optimized_scores = optimized_tc(worlds)
    assert optimized_scores.shape == (2**num_variables, 1, 1)

    assert allclose(unoptimized_scores, optimized_scores)
