import torch
from torch import autograd

from cirkit.backend.torch.semiring import ComplexLSESumSemiring, LSESumSemiring
from cirkit.backend.torch.utils import csafelog
from tests.floats import allclose


def test_complex_safelog_derivative():
    torch.set_grad_enabled(True)
    z = torch.randn(1000, dtype=torch.complex128)
    z.requires_grad = True
    assert autograd.gradcheck(csafelog, z)

    torch.set_default_dtype(torch.float32)
    z = 1.0 + 2.0 * torch.randn(1000, dtype=torch.complex64)
    z.requires_grad = True
    log_y = torch.log(z)
    log_y.mean().real.backward()
    grad_log_y = torch.clone(z.grad)
    z.grad = None
    safelog_y = csafelog(z)
    safelog_y.mean().real.backward()
    grad_safelog_y = torch.clone(z.grad)
    assert allclose(log_y, safelog_y)
    assert allclose(grad_log_y, grad_safelog_y)

    z = torch.zeros(5, dtype=torch.complex64)
    mask = [False, False, True, False, False]
    z.real[mask] = 1.0
    z.imag[mask] = 1.0
    z.real[0] = -z.real[0]
    z.requires_grad = True
    y = csafelog(z)
    y.mean().real.backward()
    assert torch.all(torch.isfinite(z.grad))


def test_complex_lse_sum_semiring():
    torch.set_default_dtype(torch.float32)

    x = torch.tensor(
        [
            [-200.0, -200.0, -5.0],
        ]
    )

    w = torch.tensor([[1.0], [2.0], [1e-38]])

    y1 = LSESumSemiring.einsum("kj,ji->ki", inputs=(x,), operands=(w,), dim=-1, keepdim=True)

    y2 = ComplexLSESumSemiring.einsum(
        "kj,ji->ki", inputs=(x.to(torch.complex64),), operands=(w,), dim=-1, keepdim=True
    )

    assert torch.all(torch.isfinite(y1))
    assert torch.all(torch.isfinite(y2.real))
    assert torch.all(torch.isfinite(y2.imag))
    assert allclose(y1, y2.real)
