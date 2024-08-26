import itertools
from typing import Optional

import pytest

from cirkit.templates.region_graph import FullyFactorized, RandomBinaryTree, RegionNode
from cirkit.utils.scope import Scope
from tests.templates.region_graph.test_utils import check_region_graph_save_load


@pytest.mark.parametrize(
    "num_variables,num_repetitions",
    itertools.product([1, 5], [1, 3]),
)
def test_rg_algorithm_fully_factorized(num_variables: int, num_repetitions: int):
    rg = FullyFactorized(num_variables, num_repetitions=num_repetitions)
    assert rg.is_omni_compatible
    root: RegionNode
    (root,) = list(rg.outputs)
    if num_variables > 1:
        assert len(rg.region_inputs(root)) == num_repetitions
    assert isinstance(root, RegionNode)
    assert root.scope == Scope(range(num_variables))
    assert all(len(rg.partition_inputs(ptn)) == num_variables for ptn in rg.partition_nodes)
    check_region_graph_save_load(rg)


@pytest.mark.parametrize(
    "num_variables,depth,num_repetitions",
    itertools.product([3, 4], [None, 1, 2], [1, 3]),
)
def test_rg_algorithm_random_binary_tree(
    num_variables: int, depth: Optional[int], num_repetitions: int
):
    rg = RandomBinaryTree(num_variables, depth=depth, num_repetitions=num_repetitions)
    if num_repetitions == 1:
        assert rg.is_structured_decomposable
    root: RegionNode
    (root,) = list(rg.outputs)
    assert isinstance(root, RegionNode)
    assert root.scope == Scope(range(num_variables))
    assert len(rg.region_inputs(root)) == num_repetitions
    assert all(len(rg.region_inputs(rgn)) == 1 for rgn in rg.inner_region_nodes)
    assert all(len(rg.partition_inputs(ptn)) == 2 for ptn in rg.partition_nodes)
    check_region_graph_save_load(rg)
    rg.dump(f"rg-{num_variables}-{depth}-{num_repetitions}.json")
