import itertools
from typing import List, Tuple, Union

import numpy as np
import pytest

from cirkit.new.region_graph.algorithms.fully_factorized import FullyFactorized

def test_rg_full() -> None:
    num_vars = 4
    rg = FullyFactorized(num_vars=num_vars)
    assert rg.is_smooth
    assert rg.is_decomposable
    if np.prod(num_vars) > 1:
        assert rg.is_structured_decomposable