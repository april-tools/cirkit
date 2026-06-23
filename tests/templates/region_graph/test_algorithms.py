import itertools

import numpy as np
import pytest
import torch

from cirkit.templates.region_graph import (
    ChowLiuTree,
    FullyFactorized,
    LinearTree,
    PoonDomingos,
    QuadGraph,
    QuadTree,
    RandomBinaryTree,
    RegionGraph,
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
    "shape,num_patch_splits",
    itertools.product(
        [(1, 1, 1), (1, 1, 3), (1, 3, 1), (1, 3, 3), (3, 3, 3), (1, 4, 4), (3, 4, 4)], [2, 4]
    ),
)
def test_rg_algorithm_quad_tree(shape: tuple[int, int], num_patch_splits: int):
    num_variables = np.prod(shape)
    rg = QuadTree(shape, num_patch_splits=num_patch_splits)
    root: RegionNode
    (root,) = list(rg.outputs)
    assert isinstance(root, RegionNode)
    assert root.scope == Scope(range(num_variables))
    assert all(len(rgn.scope) == shape[0] for rgn in rg.inputs)
    assert all(len(rg.region_inputs(rgn)) == 1 for rgn in rg.inner_region_nodes)
    if num_patch_splits == 2:
        assert all(len(rg.partition_inputs(ptn)) == 2 for ptn in rg.partition_nodes)
    else:
        if num_variables == 16:
            assert all(len(rg.partition_inputs(ptn)) == 4 for ptn in rg.partition_nodes)
        else:
            assert all(len(rg.partition_inputs(ptn)) in [2, 4] for ptn in rg.partition_nodes)
    check_region_graph_save_load(rg)


@pytest.mark.parametrize(
    "shape", [(1, 1, 1), (1, 1, 3), (1, 3, 1), (1, 3, 3), (3, 3, 3), (1, 4, 4), (3, 4, 4)]
)
def test_rg_algorithm_quad_graph(shape: tuple[int, int]):
    num_variables = np.prod(shape)
    rg = QuadGraph(shape)
    root: RegionNode
    (root,) = list(rg.outputs)
    assert isinstance(root, RegionNode)
    assert root.scope == Scope(range(num_variables))
    assert all(len(rgn.scope) == shape[0] for rgn in rg.inputs)
    assert all(len(rg.region_inputs(rgn)) in [1, 2] for rgn in rg.inner_region_nodes)
    assert all(len(rg.partition_inputs(ptn)) in [2, 4] for ptn in rg.partition_nodes)
    check_region_graph_save_load(rg)


@pytest.mark.parametrize(
    "shape,delta",
    itertools.product(
        [(1, 1, 1), (1, 3, 3), (1, 4, 4), (3, 3, 3), (3, 4, 4)], [1, [1, 2], [[1, 3], [2, 4]]]
    ),
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


def _chow_liu_tree_data(
    input_type: str, parents: list[int], is_categorical: list[bool], num_samples: int
) -> torch.Tensor:
    torch.manual_seed(42)
    coupling = 0.9
    num_categories = 4
    num_variables = len(parents)
    if input_type == "categorical":
        columns = [torch.randint(0, num_categories, (num_samples,))]
        for parent in parents[1:]:
            noise = torch.randint(0, num_categories, (num_samples,))
            # child = parent with coupling probability, else child = random noise
            mask = torch.rand(num_samples) < coupling
            columns.append(torch.where(mask, columns[parent], noise))
        return torch.stack(columns, dim=1)
    latent = [torch.randn(num_samples)]
    for parent in parents[1:]:
        noise = torch.randn(num_samples)
        latent.append(coupling * latent[parent] + (1 - coupling**2) ** 0.5 * noise)
    if input_type == "gaussian":
        return torch.stack(latent, dim=1)
    # heterogeneous data
    probs = torch.arange(1, num_categories, dtype=torch.float32) / num_categories
    cut_points = torch.distributions.Normal(0.0, 1.0).icdf(probs)
    columns = [
        torch.bucketize(latent[v], cut_points).float() if is_categorical[v] else latent[v]
        for v in range(num_variables)
    ]
    return torch.stack(columns, dim=1)


@pytest.mark.parametrize(
    "input_type,chunk_size,heter_cont_bins",
    [
        (input_type, chunk_size, heter_cont_bins)
        for input_type in ["categorical", "gaussian", "heterogeneous"]
        for chunk_size in [None, 256]
        for heter_cont_bins in ([None, 10] if input_type == "heterogeneous" else [None])
    ],
)
def test_rg_algorithm_chow_liu_tree(
    input_type: str, chunk_size: int | None, heter_cont_bins: int | None
):
    # parent index for each variable, -1 when the variable is the root
    parents = [-1, 0, 0, 1, 1, 2]
    # for heterogeneous input_type only
    is_categorical = [False, False, True, False, True, True]
    num_samples = 8000
    num_variables = len(parents)
    data = _chow_liu_tree_data(input_type, parents, is_categorical, num_samples)
    if input_type == "heterogeneous":
        rg_input_type = ["categorical" if c else "gaussian" for c in is_categorical]
    else:
        rg_input_type = input_type

    tree = ChowLiuTree(
        data=data,
        input_type=rg_input_type,
        root=0,
        chunk_size=chunk_size,
        heter_cont_bins=heter_cont_bins,
        as_region_graph=False,
    )
    assert tree.shape == (num_variables,)
    assert np.array_equal(tree, np.asarray(parents))

    rg = ChowLiuTree(
        data=data,
        input_type=rg_input_type,
        root=0,
        chunk_size=chunk_size,
        heter_cont_bins=heter_cont_bins,
        as_region_graph=True,
    )
    assert isinstance(rg, RegionGraph)
    root: RegionNode
    (root,) = list(rg.outputs)
    assert isinstance(root, RegionNode)
    assert root.scope == Scope(range(num_variables))
    assert rg.is_structured_decomposable
    assert not rg.is_omni_compatible
    assert len(list(rg.inputs)) == num_variables
    assert all(len(rgn.scope) == 1 for rgn in rg.inputs)
    assert {variable for rgn in rg.inputs for variable in rgn.scope} == set(range(num_variables))
    assert all(len(rg.partition_inputs(ptn)) >= 2 for ptn in rg.partition_nodes)
