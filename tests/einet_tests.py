from cirkit.einet.einet import _Args, LowRankEiNet
from cirkit.einet.exp_family import CategoricalArray
from cirkit.region_graph.quad_tree import QuadTree
from cirkit.einet.einsum_layer.cp_einsum_layer import CPEinsumLayer


def test_einet_creation():
    """Tests the creation of an einet."""
    device = "cpu"

    # graph = Graph.random_binary_trees(num_var=4, depth=3, num_repetitions=2)
    graph = QuadTree(4, 4, struct_decomp=True)

    args = _Args(
        rg_structure="quad_tree_stdec",
        layer_type=CPEinsumLayer,
        exponential_family=CategoricalArray,
        exponential_family_args={'K': 2},
        num_sums=16,
        num_input=16,
        num_var=16,
        prod_exp=True,
        r=1
    )

    einet = LowRankEiNet(graph, args)
    einet.initialize(exp_reparam=False, mixing_softmax=False)
    einet.to(device)

    assert True
