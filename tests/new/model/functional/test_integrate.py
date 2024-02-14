import itertools

import torch

from cirkit.new.layers import InputLayer
from tests import floats
from tests.new.model.test_prod_utils import get_two_circuits, pf_of_product_of_normal
from tests.new.model.test_utils import get_circuit_2x2, set_circuit_2x2_params


def test_circuit_part_func() -> None:
    circuit = get_circuit_2x2()
    set_circuit_2x2_params(circuit)
    all_inputs = torch.tensor(
        list(itertools.product([0, 1], repeat=4))  # type: ignore[misc]
    ).unsqueeze(
        dim=-1
    )  # shape (B=16, D=4, C=1).
    output = circuit(all_inputs)  # shape (B=16, num_out=1, num_cls=1).
    sum_output = torch.logsumexp(output, dim=0)
    part_func = circuit.partition_func
    assert floats.allclose(part_func, sum_output)
    assert floats.allclose(part_func, 0.0)


def test_circuit_product_part_func() -> None:
    (circuit1, circuit2) = get_two_circuits(same_scope=True, setting="norm")

    product_part_circuit = (circuit1.product(circuit2)).partition_circuit
    layer_input = torch.zeros(1, 1)

    for layer in product_part_circuit.layers:  # type: ignore[misc]
        if isinstance(layer, InputLayer):  # type: ignore[misc]
            pf_output_by_expression = pf_of_product_of_normal(layer.params())  # type: ignore[misc]
            pf_output_by_layer = layer.forward(layer_input)
            assert torch.allclose(pf_output_by_expression, pf_output_by_layer)
