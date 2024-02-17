import torch

from cirkit.new import set_layer_comp_space
from tests import floats
from tests.new.model.test_utils import get_circuit_2x2, set_circuit_2x2_params


def test_circuit_grad() -> None:
    set_layer_comp_space("linear")  # TODO: what happens in log-space? will log(-1) appear?
    circuit = get_circuit_2x2("norm")
    set_circuit_2x2_params(circuit, "norm")
    # shape (B=2, D=4, C=1).
    inputs = (
        torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]])  # type: ignore[misc]
        .unsqueeze(dim=-1)
        .requires_grad_()
    )
    output = circuit(inputs)  # shape (B=2, num_out=1, num_cls=1).
    # shape (B=2, D=4, C=1).
    (grad_autodiff,) = torch.autograd.grad(output, inputs, torch.ones_like(output))
    grad_circuit = circuit.grad_circuit
    grad = grad_circuit(inputs)
    assert grad.shape == (2, 5, 1)  # shape (B=2, num_out=(1, (4, 1) + 1), num_cls=1).
    grad = grad[:, :-1, :]  # shape (B=2, num_out=4, num_cls=1).
    # TODO: what if C!=1 or num_cls!=1? test?
    assert floats.allclose(grad, grad_autodiff)
    assert floats.allclose(grad[0], 0.0)  # Grad at mu of Normal should be 0.
    set_layer_comp_space("log")  # TODO: use a with to tmp set default?
