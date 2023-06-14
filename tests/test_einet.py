import itertools
from typing import Dict, List, Type, Union

import pytest
import torch
from torch import Tensor

from cirkit.layers.einsum_layer.cp_einsum_layer import CPEinsumLayer
from cirkit.layers.exp_family_input_layer import CategoricalInputLayer
from cirkit.models.einet import LowRankEiNet, _Args
from cirkit.region_graph import RegionGraph
from cirkit.region_graph.poon_domingos_structure import PoonDomingosStructure
from cirkit.region_graph.quad_tree import QuadTree
from cirkit.region_graph.random_binary_tree import RandomBinaryTree
from cirkit.utils import RandomCtx


@pytest.mark.parametrize(
    "rg_cls,kwargs,log_answer",
    [
        (PoonDomingosStructure, {"shape": [4, 4], "delta": 2}, 9.79221248626709),
        (QuadTree, {"width": 4, "height": 4, "struct_decomp": False}, 52.30215835571289),
        (RandomBinaryTree, {"num_vars": 16, "depth": 3, "num_repetitions": 2}, 24.429569244384766),
    ],
)
@RandomCtx(42)
def test_einet_partition_function(
    rg_cls: Type[RegionGraph], kwargs: Dict[str, Union[int, bool, List[int]]], log_answer: float
) -> None:
    """Tests the creation and partition of an einet.

    Args:
        rg_cls (Type[RegionGraph]): The class of RG to test.
        kwargs (Dict[str, Union[int, bool, List[int]]]): The args for class to test.
        log_answer (float): The answer of partition func. NOTE: we don't know if it's correct, but \
            it guarantees reproducibility.
    """
    # TODO: type of kwargs should be refined
    device = "cpu"

    graph = rg_cls(**kwargs)

    args = _Args(
        layer_type=CPEinsumLayer,
        exponential_family=CategoricalInputLayer,
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
    out = einet(torch.tensor(all_lists))

    # log sum exp on outputs to compute their sum
    out_max: Tensor = torch.max(out, dim=0, keepdim=True)[0]  # TODO: max typing issue in pytorch
    probs = torch.exp(out - out_max)
    total_prob = probs.sum(0)
    log_prob = torch.log(total_prob) + out_max

    assert torch.isclose(log_prob, torch.tensor(log_answer), rtol=1e-6, atol=0)
    assert torch.isclose(einet.partition_function(), log_prob, rtol=1e-6, atol=0)
