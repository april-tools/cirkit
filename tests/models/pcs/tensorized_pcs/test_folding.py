import itertools
from typing import Type

import pytest
import torch

from cirkit.layers.sum_product import (
    BaseCPLayer,
    CollapsedCPLayer,
    SharedCPLayer,
    UncollapsedCPLayer,
)
from cirkit.models.functional import integrate
from cirkit.reparams.leaf import ReparamExp, ReparamSoftmax
from cirkit.utils import RandomCtx
from tests import floats
from tests.models.pcs.tensorized_pcs.test_utils import get_pc_5_sparse


@pytest.mark.parametrize(
    "normalized,layer_cls",
    list(itertools.product([False, True], [CollapsedCPLayer, UncollapsedCPLayer, SharedCPLayer])),
)
@RandomCtx(42)
def test_pc_sparse_folding(normalized: bool, layer_cls: Type[BaseCPLayer]) -> None:
    reparam = ReparamSoftmax if normalized else ReparamExp  # TODO: pass in class instead of flag?
    pc = get_pc_5_sparse(reparam, layer_cls=layer_cls)
    assert any(should_pad for (should_pad, _, __) in pc.bookkeeping)
    assert any(not should_pad for (should_pad, _, _) in pc.bookkeeping)
    data = torch.tensor(list(itertools.product([0, 1], repeat=5))).unsqueeze(  # type: ignore[misc]
        dim=-1
    )
    pc_pf = integrate(pc)
    log_z = pc_pf(data)
    log_scores = pc(data)
    lls = log_scores - log_z
    # TODO: atol is quite large here, I think it has to do with how we
    #  initialize the parameters, and for some of them it lose precision in float32
    assert floats.allclose(torch.logsumexp(lls, dim=0), 0.0)
    if normalized:
        assert floats.allclose(log_z, 0.0)
