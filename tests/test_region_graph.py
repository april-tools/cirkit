# pylint: disable=missing-function-docstring

import itertools
import tempfile
from typing import List, Set, Tuple, Union

import numpy as np
import pytest

from cirkit.region_graph import PartitionNode, RegionGraph, RegionNode
from cirkit.region_graph.poon_domingos_structure import PoonDomingosStructure
from cirkit.region_graph.quad_tree import QuadTree
from cirkit.region_graph.random_binary_tree import RandomBinaryTree


def check_smoothness_decomposability(rg: RegionGraph):
    for r in rg.inner_region_nodes:
        iscopes: Set[Tuple] = set(tuple(n.scope) for n in rg.get_node_input(r))
        assert len(iscopes) == 1, "Smoothness is not satisfied"
        assert set(list(iscopes)[0]) == r.scope, "Inconsistent scopes"
    for p in rg.partition_nodes:
        iscopes: List[Set] = list(n.scope for n in rg.get_node_input(p))
        for i, j in itertools.combinations(range(len(iscopes)), 2):
            assert not (iscopes[i] & iscopes[j]), "Decomposability is not satisfied"
            assert iscopes[i].issubset(p.scope), "Inconsistent scopes"


def check_strong_structured_decomposability(rg: RegionGraph):
    check_smoothness_decomposability(rg)
    decomps = {}
    for p in rg.partition_nodes:
        scope = tuple(sorted(list(p.scope)))
        print(scope)
        iscopes: List[Set] = list(n.scope for n in rg.get_node_input(p))
        if scope not in decomps:
            decomps[scope] = iscopes
            continue
        oth_iscopes = set(tuple(s) for s in iscopes)
        cur_iscopes = set(tuple(s) for s in decomps[scope])
        assert oth_iscopes == cur_iscopes, "Strong structured-decomposability is not satisfied"


def check_region_partition_layers(rg: RegionGraph, bottom_up: bool):
    layers = rg.topological_layers(bottom_up)
    if not bottom_up:  # Reverse layers
        layers = layers[::-1]
    for i, layer in enumerate(layers):
        if not i % 2:  # Layer of regions at the inputs
            for r in layer:
                assert isinstance(r, RegionNode), "Inconsistent layer of regions"
        else:
            for p in layer:
                assert isinstance(p, PartitionNode), "Inconsistent layer of partitions"


def check_equivalent_region_graphs(rg1: RegionGraph, rg2: RegionGraph):
    rg1_nodes = sorted(rg1.nodes)
    rg2_nodes = sorted(rg2.nodes)
    assert len(rg1_nodes) == len(
        rg2_nodes
    ), f"Region graphs have not the same number of nodes: {len(rg1_nodes)} and {len(rg2_nodes)}"
    for n, m in zip(rg1_nodes, rg2_nodes):
        assert (
            n.__class__ == m.__class__
        ), "Region graphs have nodes with different types at the same locations"
        assert (
            n.scope == m.scope
        ), f"Region graphs have nodes with different scopes at the same locations:" \
           f" {n.scope} and {m.scope}"


def check_region_graph_save_load(rg: RegionGraph):
    with tempfile.NamedTemporaryFile("r+") as f:
        rg.save(f.name)
        f.seek(0)
        loaded_rg = RegionGraph.load(f.name)
        check_equivalent_region_graphs(rg, loaded_rg)


@pytest.mark.parametrize(
    "num_vars,depth,num_repetitions", list(itertools.product([1, 8, 16], [0, 1, 3], [1, 3]))
)
def test_rg_random_binary_tree(num_vars: int, depth: int, num_repetitions: int):
    # Not enough variables for splitting at a certain depth
    if num_vars < 2**depth:
        with pytest.raises(AssertionError):
            RandomBinaryTree(num_vars, depth, num_repetitions, random_state=42)
        return

    rg = RandomBinaryTree(num_vars, depth, num_repetitions, random_state=42)
    check_smoothness_decomposability(rg)
    check_region_partition_layers(rg, bottom_up=True)
    # This does not terminate
    # check_region_partition_layers(rg, bottom_up=False)
    check_region_graph_save_load(rg)


@pytest.mark.parametrize(
    "size,struct_decomp", list(itertools.product([(1, 1), (17, 17), (32, 32)], [False, True]))
)
def test_rg_quad_tree(size: Tuple[int, int], struct_decomp: bool):
    width, height = size
    rg = QuadTree(width, height, struct_decomp=struct_decomp)
    if struct_decomp:
        check_strong_structured_decomposability(rg)
    else:
        check_smoothness_decomposability(rg)
        if np.prod(size) > 1:
            with pytest.raises(AssertionError):
                check_strong_structured_decomposability(rg)
    check_region_partition_layers(rg, bottom_up=True)
    # This does not terminate
    # check_region_partition_layers(rg, bottom_up=False)
    check_region_graph_save_load(rg)


@pytest.mark.parametrize(
    "shape,delta",
    list(itertools.product(
        [(1, 1), (3, 3), (8, 8)],
        [1.0, [1.0, 2.0], [[1.0, 3.0], [2.0, 4.0]]]
    )))
def test_rg_poon_domingos(
    shape: Tuple[int, int], delta: Union[float, List[float], List[List[float]]]
):
    rg = PoonDomingosStructure(shape, delta)
    check_smoothness_decomposability(rg)
    check_region_partition_layers(rg, bottom_up=True)
    # This does not terminate
    # check_region_partition_layers(rg, bottom_up=False)
    check_region_graph_save_load(rg)
