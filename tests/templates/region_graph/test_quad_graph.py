import itertools
from typing import Tuple

import numpy as np
import pytest

from cirkit.templates.region_graph import QuadGraph
from tests.templates.region_graph.test_utils import check_region_graph_save_load


@pytest.mark.parametrize(
    "size,num_patch_splits", itertools.product([(1, 1), (7, 5), (15, 15), (16, 16)], [2, 4])
)
def test_rg_property_quad_tree(size: Tuple[int, int], num_patch_splits: int) -> None:
    width, height = size
    rg = QuadGraph((height, width), is_tree=True, num_patch_splits=num_patch_splits)
    if np.prod(size) > 1:
        assert rg.is_structured_decomposable
    assert rg.is_smooth
    assert rg.is_decomposable
    # TODO: save/loading not working!
    # check_region_graph_save_load(rg)


@pytest.mark.parametrize("size", [(1, 1), (7, 5), (15, 15), (16, 16)])
def test_rg_property_quad_graph(size: Tuple[int, int]) -> None:
    width, height = size
    rg = QuadGraph((height, width))
    if np.prod(size) > 1:
        assert not rg.is_structured_decomposable
    assert rg.is_smooth
    assert rg.is_decomposable
    # TODO: save/loading not working!
    # check_region_graph_save_load(rg)
