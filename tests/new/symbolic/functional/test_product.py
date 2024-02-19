from cirkit.new.region_graph import QuadTree
from tests.new.symbolic.test_utils import get_symbolic_circuit_on_rg


def test_symbolic_product_qt() -> None:
    rg = QuadTree((4, 4), struct_decomp=True)

    symb_circuit = get_symbolic_circuit_on_rg(rg, setting="norm")

    prod_circuit = symb_circuit @ symb_circuit

    pf_of_prod = prod_circuit.integrate()

    assert len(list(pf_of_prod.layers)) == 62
    assert len(list(pf_of_prod.input_layers)) == 16
    assert len(list(pf_of_prod.output_layers)) == 1
