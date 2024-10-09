import itertools

import pytest

from cirkit.templates.region_graph import (
    FullyFactorized,
    LinearTree,
    PoonDomingos,
    QuadGraph,
    QuadTree,
    RandomBinaryTree,
    RegionNode,
)
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
    assert isinstance(root, RegionNode)
    if num_variables > 1:
        assert len(rg.region_inputs(root)) == num_repetitions
    assert root.scope == Scope(range(num_variables))
    assert all(len(rg.partition_inputs(ptn)) == num_variables for ptn in rg.partition_nodes)
    check_region_graph_save_load(rg)


@pytest.mark.parametrize(
    "num_variables,num_repetitions,randomize",
    itertools.product([1, 5], [1, 3], [False, True]),
)
def test_rg_algorithm_linear_tree(num_variables: int, num_repetitions: int, randomize: bool):
    rg = LinearTree(num_variables, num_repetitions=num_repetitions, randomize=randomize)
    root: RegionNode
    (root,) = list(rg.outputs)
    assert isinstance(root, RegionNode)
    if num_variables > 1:
        assert not rg.is_omni_compatible
        assert len(rg.region_inputs(root)) == num_repetitions
    else:
        assert rg.is_omni_compatible
    assert root.scope == Scope(range(num_variables))
    assert all(len(rg.partition_inputs(ptn)) == 2 for ptn in rg.partition_nodes)
    if not randomize:
        assert all(
            rg.partition_inputs(ptn)[0].scope == Scope([sorted(ptn.scope)[0]])
            for ptn in rg.partition_nodes
        )
        assert all(
            rg.partition_inputs(ptn)[1].scope == Scope(sorted(ptn.scope)[1:])
            for ptn in rg.partition_nodes
        )
    else:
        assert all(len(rg.partition_inputs(ptn)[0].scope) == 1 for ptn in rg.partition_nodes)
        assert all(
            len(rg.partition_inputs(ptn)[1].scope) == len(ptn.scope) - 1
            for ptn in rg.partition_nodes
        )
    check_region_graph_save_load(rg)


@pytest.mark.parametrize(
    "num_variables,depth,num_repetitions",
    itertools.product([3, 4], [None, 1, 2], [1, 3]),
)
def test_rg_algorithm_random_binary_tree(
    num_variables: int, depth: int | None, num_repetitions: int
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


@pytest.mark.parametrize(
    "shape,num_patch_splits", itertools.product([(1, 1), (1, 3), (3, 1), (3, 3), (4, 4)], [2, 4])
)
def test_rg_algorithm_quad_tree(shape: tuple[int, int], num_patch_splits: int):
    num_variables = shape[0] * shape[1]
    rg = QuadTree(shape, num_patch_splits=num_patch_splits)
    root: RegionNode
    (root,) = list(rg.outputs)
    assert isinstance(root, RegionNode)
    assert root.scope == Scope(range(num_variables))
    assert all(len(rgn.scope) == 1 for rgn in rg.inputs)
    assert all(len(rg.region_inputs(rgn)) == 1 for rgn in rg.inner_region_nodes)
    if num_patch_splits == 2:
        assert all(len(rg.partition_inputs(ptn)) == 2 for ptn in rg.partition_nodes)
    else:
        if num_variables == 16:
            assert all(len(rg.partition_inputs(ptn)) == 4 for ptn in rg.partition_nodes)
        else:
            assert all(len(rg.partition_inputs(ptn)) in [2, 4] for ptn in rg.partition_nodes)
    check_region_graph_save_load(rg)


@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (3, 1), (3, 3), (4, 4)])
def test_rg_algorithm_quad_graph(shape: tuple[int, int]):
    num_variables = shape[0] * shape[1]
    rg = QuadGraph(shape)
    root: RegionNode
    (root,) = list(rg.outputs)
    assert isinstance(root, RegionNode)
    assert root.scope == Scope(range(num_variables))
    assert all(len(rgn.scope) == 1 for rgn in rg.inputs)
    assert all(len(rg.region_inputs(rgn)) in [1, 2] for rgn in rg.inner_region_nodes)
    assert all(len(rg.partition_inputs(ptn)) in [2, 4] for ptn in rg.partition_nodes)
    check_region_graph_save_load(rg)


@pytest.mark.parametrize(
    "shape,delta", itertools.product([(1, 1), (3, 3), (4, 4)], [1, [1, 2], [[1, 3], [2, 4]]])
)
def test_rg_algorithm_poon_domingos(
    shape: tuple[int, int],
    delta: int | list[int] | list[list[int]],
) -> None:
    num_variables = shape[0] * shape[1]
    rg = PoonDomingos(shape, delta=delta)
    if num_variables > 1:
        assert not rg.is_structured_decomposable
    # TODO: how to test the PoonDomingos region graph?
    check_region_graph_save_load(rg)
