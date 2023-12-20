import itertools
from typing import List, Tuple, Union

import numpy as np
import pytest

from cirkit.region_graph.poon_domingos import PoonDomingos
from tests.region_graph.test_region_graph import (
    check_region_graph_save_load,
    check_region_partition_layers,
)


@pytest.mark.parametrize(
    "shape,delta",
    list(itertools.product([(1, 1), (3, 3), (8, 8)], [1, [1, 2], [[1, 3], [2, 4]]])),
)
def test_rg_poon_domingos(
    shape: Tuple[int, int],
    delta: Union[float, int, List[Union[float, int]], List[List[Union[float, int]]]],
) -> None:
    rg = PoonDomingos(shape, delta)
    assert rg.is_smooth
    assert rg.is_decomposable
    if np.prod(shape) > 1:
        assert not rg.is_structured_decomposable
    check_region_partition_layers(rg, bottom_up=True)
    check_region_partition_layers(rg, bottom_up=False)
    check_region_graph_save_load(rg)
