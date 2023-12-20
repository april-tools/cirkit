import itertools
from typing import Tuple

import numpy as np
import pytest

from cirkit.region_graph.quad_tree import QuadTree
from tests.region_graph.test_region_graph import (
    check_region_graph_save_load,
    check_region_partition_layers,
)


@pytest.mark.parametrize(
    "size,struct_decomp", list(itertools.product([(1, 1), (17, 17), (32, 32)], [False, True]))
)
def test_rg_quad_tree(size: Tuple[int, int], struct_decomp: bool) -> None:
    width, height = size
    rg = QuadTree(width, height, struct_decomp=struct_decomp)
    if struct_decomp:
        assert rg.is_structured_decomposable
    elif np.prod(size) > 1:
        assert not rg.is_structured_decomposable
    assert rg.is_smooth
    assert rg.is_decomposable
    check_region_partition_layers(rg, bottom_up=True)
    check_region_partition_layers(rg, bottom_up=False)
    check_region_graph_save_load(rg)
