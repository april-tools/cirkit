import itertools

import numpy as np
import pytest
import torch

from cirkit.einet.einet import LowRankEiNet, _Args
from cirkit.einet.einsum_layer.cp_einsum_layer import CPEinsumLayer
from cirkit.einet.exp_family import CategoricalArray
from cirkit.region_graph.quad_tree import QuadTree
from cirkit.region_graph.random_binary_tree import RandomBinaryTree


@pytest.mark.parametrize("rg", ["random_rg", "quad_tree_stdec"], indirect=True)
def test_einet_partition_function(rg: str) -> None:
    """Tests the creation of an einet."""
    device = "cpu"

    if rg == "random_rg":
        graph = RandomBinaryTree(num_vars=16, depth=3, num_repetitions=2)
    elif rg == "quad_tree_stdec":
        graph = QuadTree(4, 4, struct_decomp=True)
    else:
        raise AssertionError("Unknown rg")

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

    # Generate all possible combinations of 16 integers from the list of possible values
    possible_values = [0, 1]
    all_lists = list(itertools.product(possible_values, repeat=16))

    # compute outputs
    out = einet(torch.tensor(np.array(all_lists)))

    # log sum exp on outputs to compute their sum
    out_max = torch.max(out, dim=0, keepdim=True)[0]
    probs = torch.exp(out - out_max)
    total_prob = probs.sum(0)
    log_prob = torch.log(total_prob) + out_max

    assert torch.isclose(einet.partition_function(), log_prob, rtol=1e-5, atol=1e-15)
