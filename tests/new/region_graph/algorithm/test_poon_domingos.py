import itertools
from typing import List, Tuple, Union

import numpy as np
import pytest

from cirkit.new.region_graph.algorithms.poon_domingos import PoonDomingos

@pytest.mark.parametrize(
    "shape,delta",
    list(itertools.product([(1, 1), (3, 3), (8, 8)], [1, [1, 2], [[1, 3], [2, 4]]])),
)
def test_rg_poon_domingos(
    shape: Tuple[int, int],
    delta: Union[float, int, List[Union[float, int]], List[List[Union[float, int]]]],
) -> None:
    rg = PoonDomingos(shape, delta=delta)
    assert rg.is_smooth
    assert rg.is_decomposable
    if np.prod(shape) > 1:
        assert not rg.is_structured_decomposable