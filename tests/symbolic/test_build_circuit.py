from cirkit.symbolic.circuit import Circuit
from cirkit.templates.region_graph import RandomBinaryTree
from tests.symbolic.test_utils import (
    categorical_layer_factory,
    dense_layer_factory,
    kronecker_layer_factory,
    mixing_layer_factory,
)


def test_build_from_region_graph():
    rg = RandomBinaryTree(12, depth=3, num_repetitions=2)
    sc = Circuit.from_region_graph(
        rg,
        num_input_units=4,
        num_sum_units=8,
        input_factory=categorical_layer_factory,
        sum_factory=dense_layer_factory,
        prod_factory=kronecker_layer_factory,
        mixing_factory=mixing_layer_factory,
    )
    assert sc.is_smooth
    assert sc.is_decomposable
    # if struct_decomp:
    # assert sc.is_structured_decomposable
    # else:
    # assert not sc.is_structured_decomposable
