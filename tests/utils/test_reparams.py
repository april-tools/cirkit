# pylint: disable=missing-function-docstring
# TODO: disable checking for docstrings for every test file in tests/

from typing import Any, Type

import torch

from cirkit.reparams.leaf import (
    ReparamClamp,
    ReparamExp,
    ReparamIdentity,
    ReparamLeaf,
    ReparamLogSoftmax,
    ReparamSoftmax,
    ReparamSquare,
)
from cirkit.utils import RandomCtx


@torch.no_grad()  # type: ignore[misc]
def _get_param(cls: Type[ReparamLeaf], dim: int = 0, **kwargs: Any) -> ReparamLeaf:
    reparam = cls((8, 16, 32), dim=dim, **kwargs)  # type: ignore[misc]
    reparam.param.normal_()
    return reparam


@RandomCtx(42)
def test_reparam_leaf() -> None:
    p = _get_param(ReparamIdentity)
    assert torch.all(p() == p.param)
    # TODO: pylint bug?
    p = _get_param(ReparamExp)  # pylint: disable=redefined-variable-type
    assert torch.allclose(p(), torch.exp(p.param))
    p = _get_param(ReparamSquare)
    assert torch.allclose(p(), torch.square(p.param))
    p = _get_param(ReparamSoftmax, dim=0)
    assert torch.allclose(torch.sum(p(), dim=0), torch.ones(()))
    p = _get_param(ReparamSoftmax, dim=-1)
    assert torch.allclose(torch.sum(p(), dim=-1), torch.ones(()))
    p = _get_param(ReparamLogSoftmax, dim=0)
    # need 3eps allowed
    assert torch.allclose(torch.logsumexp(p(), dim=0), torch.zeros(()), atol=3e-7)
    p = _get_param(ReparamLogSoftmax, dim=-1)
    assert torch.allclose(torch.logsumexp(p(), dim=-1), torch.zeros(()), atol=3e-7)
    p = _get_param(ReparamClamp, min=1e-5)  # type: ignore[misc]
    assert torch.all(p() >= 1e-5)
    p = _get_param(ReparamClamp, max=-1e-5)  # type: ignore[misc]
    assert torch.all(p() <= 1e-5)
