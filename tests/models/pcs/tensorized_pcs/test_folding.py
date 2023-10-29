# pylint: disable=missing-function-docstring
import functools
import itertools
from typing import Type

import pytest
import torch

from cirkit.layers.sum_product import CPLayer, SharedCPLayer, SumProductLayer, UncollapsedCPLayer
from cirkit.models.functional import integrate
from cirkit.utils import RandomCtx
from cirkit.utils.reparams import reparam_exp, reparam_softmax
from tests.models.pcs.tensorized_pcs.test_utils import get_pc_5_sparse


@pytest.mark.parametrize(
    "normalized,layer_cls",
    list(
        itertools.product(
            [False, True], [CPLayer, UncollapsedCPLayer, SharedCPLayer]  # type: ignore[misc]
        )
    ),
)
@RandomCtx(42)
def test_pc_sparse_folding(normalized: bool, layer_cls: Type[SumProductLayer]) -> None:
    reparam = functools.partial(reparam_softmax, dim=-2) if normalized else reparam_exp
    pc = get_pc_5_sparse(reparam, layer_cls=layer_cls)  # type: ignore[arg-type]
    assert any(should_pad for (should_pad, _, __) in pc.bookkeeping)
    assert any(not should_pad for (should_pad, _, _) in pc.bookkeeping)
    data = torch.tensor(list(itertools.product([0, 1], repeat=5)))  # type: ignore[misc]
    pc_pf = integrate(pc)
    log_z = pc_pf()
    log_scores = pc(data)
    lls = log_scores - log_z
    # TODO: atol is quite large here, I think it has to do with how we
    #  initialize the parameters, and for some of them it lose precision in float32
    assert torch.allclose(torch.logsumexp(lls, dim=0), torch.zeros(()), atol=2e-6)
    if normalized:
        assert torch.allclose(log_z, torch.zeros(()), atol=2e-6)
