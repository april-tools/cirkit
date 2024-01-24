# pylint: disable=too-many-locals
import torch

from cirkit.new import set_layer_comp_space
from tests.new.model.test_prod_utils import get_two_circuits


def test_circuit_product_same_scope() -> None:
    set_layer_comp_space("linear")  # TODO: what happens in log-space? will log(-1) appear?
    (circuit1, circuit2) = get_two_circuits(same_scope=True, setting="norm")
    inputs = torch.rand(2, 4, 1) * 10  # shape (B=2, D=4, C=1).

    output1 = circuit1(inputs)
    output2 = circuit2(inputs)
    outputs_product = output1 * output2

    product_circuit = circuit1.product(circuit2)
    product_circuit_output = product_circuit(inputs)

    assert torch.allclose(product_circuit_output, outputs_product)

    # Test categorical
    (circuit1_cat, circuit2_cat) = get_two_circuits(same_scope=True, setting="cat")
    inputs_cat = torch.randint(5, (2, 4, 1)).float()  # shape (B=2, D=4, C=1).

    output1_cat = circuit1_cat(inputs_cat)
    output2_cat = circuit2_cat(inputs_cat)
    outputs_product_cat = output1_cat * output2_cat

    product_circuit_cat = circuit1_cat.product(circuit2_cat)
    product_circuit_output_cat = product_circuit_cat(inputs_cat)

    assert torch.allclose(product_circuit_output_cat, outputs_product_cat)

    set_layer_comp_space("log")  # TODO: use a with to tmp set default?
