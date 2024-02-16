import itertools

import torch

from tests import floats
from tests.new.model.test_utils import get_circuit_2x2, set_circuit_2x2_params


def test_circuit_part_func() -> None:
    circuit = get_circuit_2x2()
    set_circuit_2x2_params(circuit)
    # shape (B=16, D=4, C=1).
    all_inputs = torch.tensor(
        list(itertools.product([0, 1], repeat=4))  # type: ignore[misc]
    ).unsqueeze(dim=-1)
    output = circuit(all_inputs)  # shape (B=16, num_out=1, num_cls=1).
    sum_output = torch.logsumexp(output, dim=0)
    part_func = circuit.partition_func
    assert floats.allclose(part_func, sum_output)
    assert floats.allclose(part_func, 0.0)
