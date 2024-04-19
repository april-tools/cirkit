import itertools
from typing import List, Tuple, Union

import numpy as np
import pytest

from cirkit.templates.region_graph import PoonDomingos
from tests.templates.region_graph.test_region_graph import (
    check_region_graph_save_load
)


@pytest.mark.parametrize(
    "shape,delta",
    list(itertools.product([(1, 1), (3, 3), (4, 4)], [1, [1, 2], [[1, 3], [2, 4]]])),
)
def test_rg_poon_domingos(
    shape: Tuple[int, int],
    delta: Union[float, List[float], List[List[float]]],
) -> None:
    rg = PoonDomingos(shape, delta=delta)
    assert rg.is_smooth
    assert rg.is_decomposable
    if np.prod(shape) > 1:
        assert not rg.is_structured_decomposable
    check_region_graph_save_load(rg)
