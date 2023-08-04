# pylint: disable=missing-function-docstring
# TODO: disable checking for docstrings for every test file in tests/

import torch

from cirkit.utils import RandomCtx
from cirkit.utils.reparams import (
    reparam_exp,
    reparam_id,
    reparam_log_softmax,
    reparam_positive,
    reparam_softmax,
    reparam_square,
)


@RandomCtx(42)
def test_reparams() -> None:
    p = torch.randn(10, 10)
    assert torch.all(p == reparam_id(p))
    assert torch.allclose(reparam_exp(p), torch.exp(p))
    assert torch.allclose(reparam_square(p), torch.square(p))
    assert torch.allclose(torch.sum(reparam_softmax(p, dim=0), dim=0), torch.ones(()))
    assert torch.allclose(torch.sum(reparam_softmax(p, dim=-1), dim=-1), torch.ones(()))
    assert torch.allclose(
        torch.logsumexp(reparam_log_softmax(p, dim=0), dim=0),
        torch.zeros(()),
        atol=torch.finfo(torch.float32).eps,
    )
    assert torch.allclose(
        torch.logsumexp(reparam_log_softmax(p, dim=-1), dim=-1),
        torch.zeros(()),
        atol=torch.finfo(torch.float32).eps,
    )
    assert torch.all(reparam_positive(p, eps=1e-5) >= 1e-5)
