from cirkit.einet.einet import LowRankEiNet, _Args
from cirkit.einet.einsum_layer.cp_einsum_layer import CPEinsumLayer
from cirkit.einet.exp_family import CategoricalArray
from cirkit.region_graph.quad_tree import QuadTree
from cirkit.region_graph.random_binary_tree import RandomBinaryTree


def test_einet_creation() -> None:
    """Tests the creation of an einet."""
    device = "cpu"

    for graph in (
        RandomBinaryTree(num_vars=16, depth=3, num_repetitions=2),
        QuadTree(4, 4, struct_decomp=True),
    ):
        args = _Args(
            rg_structure="quad_tree_stdec",
            layer_type=CPEinsumLayer,
            exponential_family=CategoricalArray,
            exponential_family_args={"k": 2},  # type: ignore[misc]
            num_sums=16,
            num_input=16,
            num_var=16,
            prod_exp=True,
            r=1,
        )

        einet = LowRankEiNet(graph, args)
        einet.initialize(exp_reparam=False, mixing_softmax=False)
        einet.to(device)
