import itertools

import torch

from cirkit.new import set_layer_comp_space
from tests import floats
from tests.new.model.test_utils import (
    get_circuit_2x2,
    get_circuit_2x2_output,
    get_circuit_2x2_param_shapes,
    set_circuit_2x2_params,
)


def test_circuit_instantiation() -> None:
    circuit = get_circuit_2x2()
    param_shapes = {name: tuple(param.shape) for name, param in circuit.named_parameters()}
    assert circuit.num_vars == 4
    assert param_shapes == get_circuit_2x2_param_shapes()


def test_circuit_output_linear() -> None:
    set_layer_comp_space("linear")
    circuit = get_circuit_2x2()
    set_circuit_2x2_params(circuit)
    # shape (B=16, D=4, C=1).
    all_inputs = torch.tensor(
        list(itertools.product([0, 1], repeat=4))  # type: ignore[misc]
    ).unsqueeze(dim=-1)
    output = circuit(all_inputs)
    assert output.shape == (16, 1, 1)  # shape (B=16, num_out=1, num_cls=1).
    assert floats.allclose(output, get_circuit_2x2_output())
    set_layer_comp_space("log")  # TODO: use a with to tmp set default?


def test_circuit_output_log() -> None:
    circuit = get_circuit_2x2()
    set_circuit_2x2_params(circuit)
    # shape (B=16, D=4, C=1).
    all_inputs = torch.tensor(
        list(itertools.product([0, 1], repeat=4))  # type: ignore[misc]
    ).unsqueeze(dim=-1)
    output = circuit(all_inputs)
    assert output.shape == (16, 1, 1)  # shape (B=16, num_out=1, num_cls=1).
    assert floats.allclose(output, get_circuit_2x2_output().log())
