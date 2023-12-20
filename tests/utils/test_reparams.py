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
from tests import floats


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
    assert floats.allclose(p(), torch.exp(p.param))
    p = _get_param(ReparamSquare)
    assert floats.allclose(p(), torch.square(p.param))
    p = _get_param(ReparamSoftmax, dim=0)
    assert floats.allclose(torch.sum(p(), dim=0), 1.0)
    p = _get_param(ReparamSoftmax, dim=-1)
    assert floats.allclose(torch.sum(p(), dim=-1), 1.0)
    p = _get_param(ReparamLogSoftmax, dim=0)
    assert floats.allclose(torch.logsumexp(p(), dim=0), 0.0)
    p = _get_param(ReparamLogSoftmax, dim=-1)
    assert floats.allclose(torch.logsumexp(p(), dim=-1), 0.0)
    p = _get_param(ReparamClamp, min=1e-5)  # type: ignore[misc]
    assert torch.all(p() >= 1e-5)
    p = _get_param(ReparamClamp, max=-1e-5)  # type: ignore[misc]
    assert torch.all(p() <= 1e-5)
