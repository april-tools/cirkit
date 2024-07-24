import torch

from cirkit.backend.torch.semiring import ComplexLSESumSemiring


def test_complex_lse_sum():
    torch.set_grad_enabled(True)

    # Check the clamping is correct
    z = torch.zeros((6, 6), dtype=torch.complex128)
    ComplexLSESumSemiring._double_zero_clamp_(z.real)
    assert torch.all(z.real != torch.zeros_like(z.real))

    # Check the gradients are not NaNs
    z = torch.zeros((6, 6), dtype=torch.complex128)
    mask = torch.rand_like(z.real) < 0.5
    z.real[mask] = 1.0
    z.imag[mask] = 1.0
    z.requires_grad = True
    ComplexLSESumSemiring._double_zero_clamp_(z.real)
    y = torch.log(z)
    y.real.mean().backward()
    assert torch.all(torch.isfinite(z.grad))
