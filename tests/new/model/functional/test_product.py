import itertools
from typing import cast

import torch

from cirkit.new.layers import InputLayer, ParameterizedConstantLayer
from cirkit.new.reparams import UnaryReparam
from tests import floats
from tests.new.model.functional.test_prod_utils import get_two_circuits, pf_of_product_of_normal


def test_circuit_product_norm() -> None:
    circuit1, circuit2 = get_two_circuits(same_scope=True, setting="norm")
    inputs = torch.randn(2, 4, 1)  # shape (B=2, D=4, C=1).

    output1 = circuit1(inputs)
    output2 = circuit2(inputs)
    outputs_product = output1 + output2  # Product is sum in log-space.

    product_circuit = circuit1 @ circuit2
    product_output = product_circuit(inputs)

    assert floats.allclose(product_output, outputs_product)

    product_part_circuit = product_circuit.partition_circuit

    # We don't know what the partition should be for the whole product circuit.
    for layer in product_part_circuit.layers:  # type: ignore[misc]
        if isinstance(layer, InputLayer):  # type: ignore[misc]
            assert isinstance(layer, ParameterizedConstantLayer)
            pf_output_by_expression = pf_of_product_of_normal(
                cast(UnaryReparam, layer.params).reparams[0]()
            )
            pf_output_by_layer = layer.forward(inputs)  # Only need a dummy input here.
            assert floats.allclose(pf_output_by_expression, pf_output_by_layer)


def test_circuit_product_cat() -> None:
    circuit1, circuit2 = get_two_circuits(same_scope=True, setting="cat")
    # shape (B=16, D=4, C=1).
    inputs = torch.tensor(
        list(itertools.product([0, 1], repeat=4))  # type: ignore[misc]
    ).unsqueeze(dim=-1)

    output1 = circuit1(inputs)
    output2 = circuit2(inputs)
    outputs_product = output1 + output2  # Product is sum in log-space.

    product_circuit = circuit1 @ circuit2
    product_output = product_circuit(inputs)

    assert floats.allclose(product_output, outputs_product)

    sum_prod_output = torch.logsumexp(product_output, dim=0)
    product_part_func = product_circuit.partition_func

    assert floats.allclose(product_part_func, sum_prod_output)
